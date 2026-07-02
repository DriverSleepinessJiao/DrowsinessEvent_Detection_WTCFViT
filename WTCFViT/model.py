import torch
import torch.nn as nn
from torchinfo import summary

class PatchEmbedding2D(nn.Module):
    def __init__(self, patch_size, d_model, in_channels=1):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        patch_dim = self.patch_h * self.patch_w * in_channels
        self.linear = nn.Linear(patch_dim, d_model)

    def forward(self, x):
        x = nn.functional.unfold(x, kernel_size=(self.patch_h, self.patch_w),
                                 stride=(self.patch_h, self.patch_w))
        x = x.transpose(1, 2)
        out = self.linear(x)
        return out

class PatchPositionEncoding2D(nn.Module):
    def __init__(self, num_row, num_col, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.row_embed = nn.Embedding(num_row, d_model // 2)
        self.col_embed = nn.Embedding(num_col, d_model // 2)
        self.num_row = num_row
        self.num_col = num_col
        self.d_model = d_model

    def forward(self, x):
        B = x.shape[0]
        device = next(self.row_embed.parameters()).device
        rows = torch.arange(self.num_row, device=device).unsqueeze(1).repeat(1, self.num_col)
        cols = torch.arange(self.num_col, device=device).unsqueeze(0).repeat(self.num_row, 1)
        row_pos = self.row_embed(rows)
        col_pos = self.col_embed(cols)
        pos = torch.cat([row_pos, col_pos], dim=-1)
        pos = pos.view(-1, self.d_model)
        out = pos.unsqueeze(0).repeat(B, 1, 1)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1)
        x = x + self.dropout(attn_out)
        x2 = self.norm2(x)
        ff_out = self.ff(x2)
        out = x + self.dropout(ff_out)
        return out

class ViTEncoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, n_head, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride, 'same', groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FusionConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.fusionconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        out = self.fusionconv(x)
        return out

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, num_classes)
        )
        self.branch1 = nn.Sequential(
            DSConv(64, 32, 3),
            nn.ReLU(),
            DSConv(32, 64, 3),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            DSConv(64, 32, 5),
            nn.ReLU(),
            DSConv(32, 64, 5),
            nn.ReLU()
        )
        self.fusionconv = FusionConv(64 * 3, 64, 5)

    def forward(self, x):
        B, L, C = x.shape
        num_col = 10
        num_row = L // num_col
        x = x.view(B, num_row, num_col, C).permute(0, 3, 1, 2).contiguous()
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.fusionconv(x)
        out = self.fc(x)
        return out

class WTCF(nn.Module):
    def __init__(self, input_shape, patch_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.num_row = input_shape[0] // patch_size[0]
        self.num_col = input_shape[1] // patch_size[1]
        self.patch_embed = PatchEmbedding2D(patch_size=patch_size, d_model=d_model)
        self.pos_embed = PatchPositionEncoding2D(num_row=self.num_row, num_col=self.num_col, d_model=d_model)
        self.encoder = ViTEncoder(d_model=d_model, n_head=num_heads, num_layers=num_layers, dropout=0.1)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed(x)
        x = self.encoder(x)
        out = self.decoder(x)
        return out