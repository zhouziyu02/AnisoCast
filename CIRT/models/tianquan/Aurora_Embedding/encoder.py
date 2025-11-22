"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta
# from turtle import st
from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn
from datetime import datetime

from .fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    levels_expansion,
    pos_expansion,
    scale_expansion,
)

from .decoder import Perceiver3DDecoder
from .PatchEmbedding import LevelPatchEmbed
from .Perceiver import MLP, PerceiverResampler
from .PosEncoding import pos_scale_enc
from .util import (
    check_lat_lon_dtype,
    init_weights,
)



__all__ = ["Perceiver3DEncoder"]


class Perceiver3DEncoder(nn.Module):
    """Multi-scale multi-source multi-variable encoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: Tuple[str, str],
        static_vars: Union[Tuple[str, str], None],
        atmos_vars: Tuple[str, str],
        atmos_levels: Tuple[int,],
        patch_size: int = 4,
        latent_levels: int = 8,
        embed_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 64,
        drop_rate: float = 0.1,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        max_history_size: int = 2,
        perceiver_ln_eps: float = 1e-5,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            static_vars (tuple[str, ...], optional): All supported static variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            latent_levels (int): Number of latent pressure levels. Defaults to `8`.
            embed_dim (int, optional): Embedding dim. used in the aggregation blocks. Defaults
                to `1024`.
            num_heads (int, optional): Number of attention heads used in aggregation blocks.
                Defaults to `16`.
            head_dim (int, optional): Dimension of attention heads used in aggregation blocks.
                Defaults to `64`.
            drop_rate (float, optional): Drop out rate for input patches. Defaults to `0.1`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `2`.
            mlp_ratio (float, optional): Ratio of hidden dimensionality to embedding dimensionality
                for MLPs. Defaults to `4.0`.
            max_history_size (int, optional): Maximum number of history steps to consider. Defaults
                to `2`.
            perceiver_ln_eps (float, optional): Epsilon value for layer normalisation in the
                Perceiver. Defaults to 1e-5.
        """
        super().__init__()

        self.surf_vars = surf_vars
        self.static_vars = static_vars
        self.atmos_vars = atmos_vars
        self.atmos_levels = atmos_levels

        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # We treat the static variables as surface variables in the model.
        surf_vars = surf_vars + static_vars if static_vars is not None else surf_vars

        # Latent tokens
        assert latent_levels > 1, "At least two latent levels are required."
        self.latent_levels = latent_levels
        # One latent level will be used by the surface level.
        self.atmos_latents = nn.Parameter(torch.randn(latent_levels - 1, embed_dim))

        # Learnable embedding to encode the surface level.
        self.surf_level_encoding = nn.Parameter(torch.randn(embed_dim))
        self.surf_mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.surf_norm = nn.LayerNorm(embed_dim)

        # Position, scale, and time embeddings
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)
        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)

        # Patch embeddings
        assert max_history_size > 0, "At least one history step is required."
        self.surf_token_embeds = LevelPatchEmbed(
            surf_vars,
            patch_size,
            embed_dim,
            max_history_size,
        )
        self.atmos_token_embeds = LevelPatchEmbed(
            atmos_vars,
            patch_size,
            embed_dim,
            max_history_size,
        )

        # Learnable pressure level aggregation
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            drop=drop_rate,
            mlp_ratio=mlp_ratio,
            ln_eps=perceiver_ln_eps,
        )

        # Drop patches after encoding.
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.apply(init_weights)

        # Initialize the latents like in the Huggingface implementation of the Perceiver:
        #
        #   https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/perceiver/modeling_perceiver.py#L628
        #
        torch.nn.init.trunc_normal_(self.atmos_latents, std=0.02)
        torch.nn.init.trunc_normal_(self.surf_level_encoding, std=0.02)

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pressure level information.

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
        B, _, L, _ = x.shape
        latents = self.atmos_latents.to(dtype=x.dtype)
        latents = latents.unsqueeze(1).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_A, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, C_A, D)

        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x

    def forward(self, x_surf: torch.Tensor, x_static: torch.Tensor, x_atmos: torch.Tensor, lead_time: timedelta, lat: torch.Tensor, lon: torch.Tensor, hours: Tuple[int,]) -> torch.Tensor:
        """Peform encoding.

        Args:
            batch (:class:`.Batch`): Batch to encode.
            lead_time (timedelta): Lead time.

        Returns:
            torch.Tensor: Encoding of shape `(B, L, D)`.
        """

        B, T, _, C, H, W = x_atmos.size()
        assert x_surf.shape[:2] == (B, T), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."

        if self.static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = x_static.expand((B, T, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=2)  # (B, T, V_S + V_Static, H, W)
            surf_vars = self.surf_vars + self.static_vars

        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # Patch embed the surface level.
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype  # When using mixed precision, we need to keep track of the dtype.

        # Patch embed the atmospheric levels.
        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")
        x_atmos = self.atmos_token_embeds(x_atmos, self.atmos_vars)
        x_atmos = rearrange(x_atmos, "(b c) l d -> b c l d", b=B, c=C)

        # Add surface level encoding. This helps the model distinguish between surface and
        # atmospheric levels.
        x_surf = x_surf + self.surf_level_encoding[None, None, :].to(dtype=dtype)
        # Since the surface level is not aggregated, we add a Perceiver-like MLP only.
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        # Add atmospheric pressure encoding of shape (C_A, D) and subsequent embedding.
        atmos_levels_tensor = torch.tensor(self.atmos_levels, device=x_atmos.device)
        atmos_levels_encode = levels_expansion(atmos_levels_tensor, self.embed_dim).to(dtype=dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)[None, :, None, :]
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # Aggregate over pressure levels.
        x_atmos = self.aggregate_levels(x_atmos)  # (B, C_A, L, D) to (B, C, L, D)

        # Concatenate the surface level with the amospheric levels.
        x = torch.cat((x_surf.unsqueeze(1), x_atmos), dim=1)

        # Add position and scale embeddings to the 3D tensor.
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )
        # Encodings are (L, D).
        pos_encode = self.pos_embed(pos_encode[None, None, :].to(dtype=dtype))
        scale_encode = self.scale_embed(scale_encode[None, None, :].to(dtype=dtype))
        x = x + pos_encode + scale_encode

        # Flatten the tokens.
        x = x.reshape(B, -1, self.embed_dim)  # (B, C + 1, L, D) to (B, L', D)

        # Add lead time embedding.
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=x.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + lead_time_emb.unsqueeze(1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding.
        absolute_times_list = [t.timestamp() / 3600 for t in hours]  # Times in hours
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=x.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        x = x + absolute_time_embed.unsqueeze(1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x)
        return x


if __name__ == "__main__":
    surf_x = torch.randn((1, 2, 2, 128, 256)) # b, t, v, h, w
    atmo_x = torch.randn((1, 2, 5, 4, 128, 256)) # b, t, v, c, h, w
    static_x = torch.randn((3, 128, 256))
    surf_vars = tuple(["2m_temperature", "wind_speed_10m"])
    atmo_vars = tuple([
        "geopotential",
        "wind_speed",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ])
    static_vars = tuple([
        'lsm',
        'slt',
        'orography'
    ])

    atmo_level = tuple([500, 700, 850, 1000])

    encoder = Perceiver3DEncoder(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmo_vars,
        atmos_levels=atmo_level,
        patch_size=4,
        latent_levels=8,
        embed_dim=1024,
        num_heads=16,
        head_dim=64,
        drop_rate=0.1,
        depth=2,

    )
    lead_time = timedelta(hours=1)

    lat = torch.linspace(-90, 90, 128)
    # print(lat)
    lon = torch.linspace(0, 360, 256)
    # lat = torch.randint(-90, 90, (128,))
    # lon = torch.randint(0, 360, (256,))

    hours = [datetime(2022, 1, 1, 12, 0)]

    x = encoder(surf_x, static_x, atmo_x, lead_time, lat, lon, hours)

    print(x.shape)

    decoder = Perceiver3DDecoder(
        surf_vars=surf_vars,
        atmos_vars=atmo_vars,
        atmos_levels=atmo_level,
        patch_size=4,
        embed_dim=1024,
        depth=2,
        head_dim=64,
        num_heads=8,
        mlp_rtupleatio=4,
        drop_rate=0.1,
    )

    patch_res = tuple([8, 128//4, 256//4])

    surf_preds, atmos_preds = decoder(x, lat, lon, patch_res, lead_time)
    print(surf_preds.shape, atmos_preds.shape)

# AuroraSmall = partial(
#     Aurora,
#     encoder_depths=(2, 6, 2),
#     encoder_num_heads=(4, 8, 16),
#     decoder_depths=(2, 6, 2),
#     decoder_num_heads=(16, 8, 4),
#     embed_dim=256,
#     num_heads=8,
#     use_lora=False,
# )

# AuroraHighRes = partial(
#     Aurora,
#     patch_size=10,
#     encoder_depths=(6, 8, 8),
#     decoder_depths=(8, 8, 6),
# )