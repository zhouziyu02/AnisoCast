import math
from typing import List

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Simple patch embedding using strided convolutions."""

    def __init__(
        self,
        img_size: List[int],
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            raise ValueError(
                f"img_size {img_size} must be divisible by patch_size {patch_size}"
            )
        self.grid_size = (
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure spatial dims are at least as large as expected before cropping
        h, w = self.img_size
        if x.size(-2) < h or x.size(-1) < w:
            raise ValueError(
                f"Input spatial size {(x.size(-2), x.size(-1))} "
                f"is smaller than configured img_size {self.img_size}"
            )
        x = x[:, :, :h, :w]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerModel(nn.Module):
    """Vanilla Transformer encoder for S2S forecasting."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        img_size: List[int],
        patch_size: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pred_len: int = 2,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.pred_len = pred_len
        self.output_size = output_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_size,
            embed_dim=embed_dim,
        )
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=self.patch_embed.num_patches,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        patch_area = patch_size * patch_size
        self.head = nn.Linear(
            embed_dim,
            pred_len * output_size * patch_area,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        tokens = self.pos_encoding(tokens)
        tokens = self.dropout(tokens)
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        patch_outputs = self.head(encoded)
        preds = self._unpatchify(patch_outputs)
        return preds

    def _unpatchify(self, patch_outputs: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, _ = patch_outputs.shape
        grid_h = self.img_size[0] // self.patch_size
        grid_w = self.img_size[1] // self.patch_size
        patch_area = self.patch_size * self.patch_size

        if num_tokens != grid_h * grid_w:
            raise ValueError(
                f"Token count {num_tokens} does not match grid "
                f"{grid_h}x{grid_w}"
            )

        out = patch_outputs.view(
            bsz,
            grid_h,
            grid_w,
            self.pred_len,
            self.output_size,
            self.patch_size,
            self.patch_size,
        )
        out = out.permute(0, 3, 4, 1, 5, 2, 6).contiguous()
        out = out.view(
            bsz,
            self.pred_len,
            self.output_size,
            self.img_size[0],
            self.img_size[1],
        )
        return out

