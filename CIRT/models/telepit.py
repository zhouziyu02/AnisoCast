import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn


def _ensure_external_path(external_repo_root: str) -> None:
    """Ensure the external TelePiT repo path is importable.

    Args:
        external_repo_root: Filesystem path to the TelePiT project root that contains `S2S/models/TelePiT.py`.
    """
    if external_repo_root and external_repo_root not in sys.path:
        sys.path.append(external_repo_root)


class Model(nn.Module):
    """Wrapper to load and use TelePiT from an external repository.

    Parameters mirror the external TelePiT `Model` where relevant and
    default to values compatible with the local training pipeline.
    """

    def __init__(
        self,
        img_size: List[int] = [121, 240],
        input_size: int = 63,
        output_size: int = 63,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        wavelet_levels: int = 3,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        # Optional explicit path to TelePiT repo root; if None, defaults to user's Desktop path
        external_repo_root: Optional[str] = \
            "/Users/bytedance/Desktop/TelePiT"
    ) -> None:
        super().__init__()

        # Make sure external TelePiT repo is importable
        _ensure_external_path(external_repo_root)

        # Import here after path setup to avoid import-time failures
        from S2S.models.TelePiT import Model as ExternalTelePiT

        # Instantiate external model with passed-through args
        self._impl = ExternalTelePiT(
            img_size=img_size,
            input_size=input_size,
            output_size=output_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            wavelet_levels=wavelet_levels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


