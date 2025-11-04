import os
import importlib.util
from typing import List, Optional

import torch
import torch.nn as nn


def _load_external_telepit(external_repo_root: str):
    """Load TelePiT external module from an explicit file path to avoid name collisions."""
    telepit_file = os.path.join(external_repo_root, "S2S", "models", "TelePiT.py")
    if not os.path.isfile(telepit_file):
        return None
    spec = importlib.util.spec_from_file_location("external_telepit_module", telepit_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Model"):
        return None
    return module.Model


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

        # Try load external TelePiT; if unavailable, fall back to bundled internal implementation
        ExternalTelePiT = _load_external_telepit(external_repo_root)
        if ExternalTelePiT is None:
            from .telepit_internal import Model as InternalTelePiT
            Impl = InternalTelePiT
        else:
            Impl = ExternalTelePiT

        self._impl = Impl(
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


