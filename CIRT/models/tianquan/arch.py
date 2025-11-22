# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import lru_cache
import os
from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
try:
    from CIRT.utils.tianquan.metrics import crps
except ImportError:
    # Fallback if metrics not available
    crps = None
from timm.models.vision_transformer import trunc_normal_
import xarray as xr
try:
    from CIRT.utils.tianquan.pos_embed import get_1d_sincos_pos_embed_from_grid
except ImportError:
    # Fallback implementation
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        import numpy as np
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        return emb
import pandas as pd

from .UD_ViT import PEDadd_UViT
from .FusionModule import default_conv, SCA, DEConv, FGA
from .Aurora_Embedding.encoder import Perceiver3DEncoder
from .Aurora_Embedding.decoder import Perceiver3DDecoder


class TianQuan(nn.Module):
    """
    ResCast sub-seasonal-to-seasonal forecasting model.

    Args:
        default_vars (list[str]): List of variable names used as inputs.
        root_dir (str): Directory to save outputs (e.g., predictions).
        img_size (tuple[int,int]): Spatial resolution of input (H, W).
        patch_size (int): Size of each patch for embedding.
        embed_dim (int): Dimension of embedding space.
        decoder_depth (int): Number of decoder layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        drop_path (float): DropPath rate.
        drop_rate (float): Dropout rate.
    """
    def __init__(
        self,
        default_vars,
        root_dir,
        img_size=(128, 256),
        patch_size=2,
        embed_dim=384,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.default_vars = default_vars
        self.root_dir = root_dir

        # Define atmospheric pressure levels and variable lists
        self.levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        self.upper_variables = [
            "geopotential",
            "wind_speed",
            "temperature",
            "relative_humidity",
            "specific_humidity"
        ]
        self.surface_variables = (
            ["2m_temperature", "wind_speed_10m"]
            if img_size == (128, 256)
            else ["2m_temperature", "10m_wind_speed"]
        )
        self.static_vars = ['lsm', 'slt']

        # Level weighting normalized sum-to-one
        level_weight = [lvl / sum(self.levels) for lvl in self.levels]
        self.weight = level_weight

        # Load static constants (land mask, soil type, topography)
        self.constants = self.load_constants(os.path.join(root_dir, "constants"))

        # Data fusion and anomaly branches
        self.anomaly_decoder = DEConv(default_conv, len(default_vars), 3)
        self.data_decoder    = SCA(default_conv, len(default_vars), 3)
        self.fusion_encoder  = FGA(len(default_vars))

        # Perceiver encoder & decoder
        self.encoder = Perceiver3DEncoder(
            surf_vars=tuple(self.surface_variables),
            static_vars=tuple(self.static_vars),
            atmos_vars=tuple(self.upper_variables),
            atmos_levels=tuple(self.levels),
            patch_size=patch_size,
            latent_levels=8,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=64,
            drop_rate=drop_rate,
            depth=4,
            mlp_ratio=mlp_ratio
        )
        self.decoder = Perceiver3DDecoder(
            surf_vars=tuple(self.surface_variables),
            atmos_vars=tuple(self.upper_variables),
            atmos_levels=tuple(self.levels),
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=4,
            num_heads=6,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate
        )

        # Variable embedding
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # UViT backbone
        self.backbone = PEDadd_UViT(embed_dim=embed_dim, qk_scale=True)

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize positional embeddings and layer weights."""
        pos = get_1d_sincos_pos_embed_from_grid(
            self.var_embed.shape[-1],
            np.arange(len(self.default_vars))
        )
        self.var_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier/truncated-normal init for Linear & LayerNorm."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def load_constants(self, path):
        """
        Load static constant fields (land_mask, soil_type, orography).
        Expects NetCDF files constants_1.40625.nc or constants_5.625.nc.
        If files don't exist, creates dummy constants.
        """
        fname = (
            "constants_1.40625.nc"
            if self.img_size == (128, 256)
            else "constants_5.625.nc"
        )
        constants_path = os.path.join(path, fname)
        
        if os.path.exists(constants_path):
            try:
                ds = xr.open_dataset(constants_path)
                lsm = torch.from_numpy(ds['lsm'].values.astype(np.float32))
                slt = torch.from_numpy(ds['slt'].values.astype(np.float32))
                orog = torch.from_numpy(ds['orography'].values.astype(np.float32))
                return torch.stack([lsm, slt, orog], dim=0)
            except Exception as e:
                print(f"Warning: Failed to load constants from {constants_path}: {e}")
                print("Using dummy constants instead.")
        
        # Create dummy constants if file doesn't exist
        h, w = self.img_size
        lsm = torch.zeros(h, w, dtype=torch.float32)
        slt = torch.zeros(h, w, dtype=torch.float32)
        orog = torch.zeros(h, w, dtype=torch.float32)
        return torch.stack([lsm, slt, orog], dim=0)

    def create_var_embedding(self, dim):
        """Create learnable embeddings for each variable."""
        emb = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        var_map = {v: i for i, v in enumerate(self.default_vars)}
        return emb, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        """Map variable names to their embedding indices."""
        ids = np.array([self.var_map[v] for v in vars])
        return torch.from_numpy(ids).to(device)

    def hours_to_dates(self, hours, start_date='1979-01-01T00:00:00'):
        """
        Convert forecast hours back to datetime objects.
        Args:
            hours (torch.Tensor): Hours since reference.
            start_date (str): ISO timestamp of reference.
        Returns:
            List[pd.Timestamp]
        """
        base = pd.Timestamp(start_date)
        deltas = pd.to_timedelta(hours.cpu().numpy(), unit='h')
        return (base + deltas).tolist()

    def forward_encoder(self, x, anomaly, hours, variables, lats, lons, region_info, lead_time):
        """
        Encode inputs via anomaly & data branches, fuse, Perceiver, and UViT backbone.
        Returns tuple of surface and atmospheric predictions.
        """
        b, t, v, h, w = x.size()
        # Merge time & variable dims
        x_flat = rearrange(x, "b t v h w -> (b t) v h w")
        anom_flat = rearrange(anomaly, "b t v h w -> (b t) v h w")

        a = self.anomaly_decoder(anom_flat)
        d = self.data_decoder(x_flat)
        fused = self.fusion_encoder(a, d)
        fused = rearrange(fused, "(b t) v h w -> b t v h w", t=t)

        # Split surface vs atmosphere
        atmo = fused[:,:,2:].reshape(b, t, len(self.upper_variables), len(self.levels), h, w)
        surf = fused[:,:,:2].reshape(b, t, len(self.surface_variables), h, w)

        enc = self.encoder(surf, self.constants.to(x.device), atmo,
                           lead_time, lats, lons,
                           self.hours_to_dates(hours))
        patches = (
            self.encoder.latent_levels,
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size
        )
        ub = self.backbone(enc, use_diffusion=False)
        surf_pred, atmo_pred = self.decoder(ub, lats, lons, patches, lead_time)
        return surf_pred, atmo_pred

    def forward(self, x, y, anomaly, hours, variables, out_vars,
                lats, lons, metric, lat, region_info, lead_time):
        """
        Full forward pass returning (loss_list, predictions).
        """
        b, t, v, h, w = x.size()
        surf_pred, atmo_pred = self.forward_encoder(
            x, anomaly, hours, variables, lats, lons, region_info, lead_time
        )
        preds = torch.cat([surf_pred, atmo_pred.reshape(b, -1, h, w)], dim=1)

        # Select only output variables and region slice
        out_ids = self.get_var_ids(tuple(out_vars), x.device)
        mi, ma = region_info['min_h'], region_info['max_h']
        mj, mj = region_info['min_w'], region_info['max_w']
        y_reg = y[:, :, mi:ma+1, mj:mj+1]
        p_reg = preds[:, out_ids, mi:ma+1, mj:mj+1]

        losses = None
        if metric:
            losses = [m(p_reg, y_reg, out_vars, lat, weight=self.weight)
                      for m in metric]
        return losses, p_reg

    def evaluate(self, x, y, anomaly, hours, variables, out_vars,
                 transform, metrics, lats, lons, lat, clim, log_postfix,
                 region_info, partition, lead_time=None):
        """
        Compute evaluation metrics (excluding CRPS) on a batch.
        """
        _, preds = self.forward(
            x, y, anomaly, hours, variables, out_vars,
            lats, lons, metric=None, lat=lat,
            region_info=region_info, lead_time=lead_time
        )
        mi, ma = region_info['min_h'], region_info['max_h']
        mj, mj = region_info['min_w'], region_info['max_w']
        y_reg = y[:, :, mi:ma+1, mj:mj+1]
        lat_reg = lat[mi:ma+1]
        clim_reg = clim[:, mi:ma+1, mj:mj+1]

        return [
            m(preds, y_reg, transform, out_vars, lat_reg, clim_reg, log_postfix)
            for m in metrics if m is not crps
        ]

    def evaluate_ensemble(self, x, y, anomaly, hours, variables, out_vars,
                          transform, metrics, lats, lons, lat, clim,
                          log_postfix, region_info, partition):
        """
        Ensemble evaluation via Gaussian perturbations.
        """
        all_preds = []
        for i in range(100):
            x_in = x if i == 0 else (x + 0.3 * torch.randn_like(x))
            _, p = self.forward(
                x_in, y, anomaly, hours, variables, out_vars,
                lats, lons, metric=None, lat=lat,
                region_info=region_info
            )
            all_preds.append(p)
        stack = torch.stack(all_preds, dim=1)
        mean_pred = stack.mean(dim=1)

        mi, ma = region_info['min_h'], region_info['max_h']
        mj, mj = region_info['min_w'], region_info['max_w']
        y_reg = y[:, :, mi:ma+1, mj:mj+1]
        lat_reg = lat[mi:ma+1]
        clim_reg = clim[:, mi:ma+1, mj:mj+1]

        out = []
        for m in metrics:
            if m is crps:
                out.append(m(stack, y_reg, transform, out_vars, lat_reg, clim_reg, log_postfix))
            else:
                out.append(m(mean_pred, y_reg, transform, out_vars, lat_reg, clim_reg, log_postfix))
        return out

    def predict(self, x, anomaly, hours, variables, out_vars,
                transform, lats, lons, region_info, max_predict_range):
        """
        Generate final NetCDF predictions. Saves via `np2nc_preds`.
        """
        surf_p, atmo_p = self.forward_encoder(
            x, anomaly, hours, variables, lats, lons, region_info
        )
        preds = x.clone()
        # Map upper & surface predictions back into full variable stack
        for i, var in enumerate(self.upper_variables):
            for j, lvl in enumerate(self.levels):
                idx = self.get_var_ids((f"{var}_{lvl}",), x.device)
                preds[:, idx] = atmo_p[:, j, i:i+1]
        for i, var in enumerate(self.surface_variables):
            idx = self.get_var_ids((var,), x.device)
            preds[:, idx] = surf_p[:, i:i+1]

        preds = transform(preds)
        out_ids = self.get_var_ids(tuple(out_vars), x.device)
        mi, ma = region_info['min_h'], region_info['max_h']
        mj, mj = region_info['min_w'], region_info['max_w']
        final = preds[:, out_ids, mi:ma+1, mj:mj+1].cpu().numpy()

        return final
