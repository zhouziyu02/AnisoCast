import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Lightweight Conv-BN-Activation block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Model(nn.Module):
    """
    Minimal convolutional baseline.

    Expects inputs with shape [batch, input_size, height, width] and produces
    predictions shaped [batch, pred_len, output_size, height, width].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        pred_len: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_size = output_size

        self.encoder = nn.Sequential(
            ConvBlock(input_size, hidden_dim, kernel_size=5),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3),
        )

        self.head = nn.Conv2d(hidden_dim, output_size * pred_len, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.head(feats)
        b, _, h, w = logits.shape
        preds = logits.view(b, self.pred_len, self.output_size, h, w)
        return preds

