# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
import os

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from .arch import TianQuan
try:
    from CIRT.utils.tianquan.lr_scheduler import LinearWarmupCosineAnnealingLR
    from CIRT.utils.tianquan.metrics import (
        rmse,
        lat_weighted_acc,
        lat_weighted_mse,
        lat_weighted_mse_val,
        lat_weighted_rmse, 
        crps
    )
except ImportError:
    # Fallback: create dummy functions if utils not available
    def LinearWarmupCosineAnnealingLR(*args, **kwargs):
        return None
    def rmse(*args, **kwargs):
        return {}
    def lat_weighted_acc(*args, **kwargs):
        return {}
    def lat_weighted_mse(*args, **kwargs):
        return {}
    def lat_weighted_mse_val(*args, **kwargs):
        return {}
    def lat_weighted_rmse(*args, **kwargs):
        return {}
    def crps(*args, **kwargs):
        return {}


class SubseasonalForecastModule(LightningModule):
    """Lightning module for subseasonal forecasting with the TianQuan model.

    Args:
        net (TianQuan): TianQuan model.
        pretrained_path (str, optional): Path to pretrained checkpoint.
        model_name (str, optional): Identifier for logging.
        lr (float): Learning rate.
        beta_1 (float): AdamW beta1.
        beta_2 (float): AdamW beta2.
        weight_decay (float): Weight decay for AdamW.
        warmup_epochs (int): Number of warmup steps.
        max_epochs (int): Total number of training steps.
        warmup_start_lr (float): Starting learning rate for warmup.
        eta_min (float): Minimum learning rate after annealing.
    """

    def __init__(
        self,
        net: TianQuan,
        pretrained_path: str = "",
        model_name: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if pretrained_path and os.path.exists(pretrained_path):
            if pretrained_path.endswith(".ckpt"):
                self.load_pretrained_weights(pretrained_path)
            else:
                self.load_pretrained_pth_weights(pretrained_path)
        self.ddeg_out = os.path.basename(pretrained_path)[:-5]

    def load_pretrained_pth_weights(self, pretrained_path: str):
        """Load a .pth checkpoint mapping its 'model' keys under 'net.'."""
        state = torch.load(pretrained_path, map_location="cpu")["model"]
        updated = {f"net.{k}": v for k, v in state.items()}
        print(f"Loading pretrained .pth checkpoint from: {pretrained_path}")
        msg = self.load_state_dict(updated, strict=False)
        print(msg)

    def load_pretrained_weights(self, pretrained_path: str):
        """Load a PyTorch Lightning .ckpt checkpoint, remapping any 'channel' keys."""
        if pretrained_path.startswith("http"):
            ckpt = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            ckpt = torch.load(pretrained_path, map_location="cpu")
        print(f"Loading pretrained .ckpt checkpoint from: {pretrained_path}")

        ckpt_model = ckpt["state_dict"]
        current = self.state_dict()

        # Rename any 'channel' keys to 'var'
        for k in list(ckpt_model):
            if "channel" in k:
                ckpt_model[k.replace("channel", "var")] = ckpt_model.pop(k)

        # Remove mismatched or unexpected keys
        for k in list(ckpt_model):
            if k not in current or ckpt_model[k].shape != current[k].shape:
                print(f"Removing key {k} from checkpoint")
                ckpt_model.pop(k)

        msg = self.load_state_dict(ckpt_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        """Configure standard mean–std denormalization."""
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        """Store latitude and longitude arrays for weighted metrics."""
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r: int):
        """Store the forecast range (in timesteps)."""
        self.pred_range = r

    def set_val_clim(self, clim):
        """Store validation climatology for anomaly metrics."""
        self.val_clim = clim

    def set_test_clim(self, clim):
        """Store test climatology for anomaly metrics."""
        self.test_clim = clim

    def get_patch_size(self):
        """Return the model's patch size."""
        return self.net.patch_size

    def training_step(self, batch: Any, batch_idx: int):
        x, y, anomaly, abs_hours, variables, out_variables, lats, lons, region_info, lead_time = batch

        loss_dict, _ = self.net.forward(
            x, y, anomaly, abs_hours, variables, out_variables,
            lats, lons,
            metrics=[lat_weighted_mse],
            lat=self.lat,
            region_info=region_info,
            lead_time=lead_time
        )
        loss_dict = loss_dict[0]

        # Log each variable's loss
        for var_name, var_loss in loss_dict.items():
            self.log(f"train/{var_name}", var_loss, on_step=True, prog_bar=True)

        return loss_dict["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, anomaly, abs_hours, variables, out_variables, lats, lons, region_info, lead_time = batch

        hours_total = self.pred_range * 6
        if hours_total < 24:
            postfix = f"{self.pred_range}_hours"
        else:
            postfix = f"{hours_total // 24}_days"

        metrics_out = self.net.evaluate(
            x, y, anomaly, abs_hours, variables, out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, rmse, lat_weighted_acc, crps],
            lats=lats, lons=lons, lat=self.lat,
            clim=self.val_clim,
            log_postfix=postfix,
            region_info=region_info,
            partition='val',
            lead_time=lead_time
        )

        # Flatten and log
        flat = {k: v for d in metrics_out for k, v in d.items()}
        for name, val in flat.items():
            self.log(f"val/{name}", val, on_epoch=True, sync_dist=True)

        return flat

    def test_step(self, batch: Any, batch_idx: int):
        x, y, anomaly, abs_hours, variables, out_variables, lats, lons, region_info, lead_time = batch

        hours_total = self.pred_range * 6
        if hours_total < 24:
            postfix = f"{hours_total}_hours"
        else:
            postfix = f"{hours_total // 24}_days"

        metrics_out = self.net.evaluate(
            x, y, anomaly, abs_hours, variables, out_variables,
            transform=self.denormalization,
            metrics=[rmse, lat_weighted_acc, crps],
            lats=lats, lons=lons, lat=self.lat,
            clim=self.test_clim,
            log_postfix=postfix,
            region_info=region_info,
            partition='test',
            lead_time=lead_time
        )

        flat = {k: v for d in metrics_out for k, v in d.items()}
        for name, val in flat.items():
            self.log(f"test/{name}", val, on_step=True, on_epoch=True, sync_dist=True)

        return flat

    def predict_step(self, batch: Any, batch_idx: int):
        x, anomaly, abs_hours, variables, out_variables, lats, lons, region_info = batch

        hours_total = self.pred_range * 6
        if hours_total < 24:
            postfix = f"{hours_total}_hours"
        else:
            postfix = f"{hours_total // 24}_days"

        preds = self.net.predict(
            x, anomaly, abs_hours, variables, out_variables,
            transform=self.denormalization,
            lats=lats, lons=lons,
            region_info=region_info,
            max_predict_range=hours_total
        )
        print(preds.shape)
        return preds

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if any(key in name for key in ("var_embed", "pos_embed", "time_pos_embed")):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay,     "lr": self.hparams.lr, "betas": (self.hparams.beta_1, self.hparams.beta_2), "weight_decay": self.hparams.weight_decay},
            {"params": no_decay,  "lr": self.hparams.lr, "betas": (self.hparams.beta_1, self.hparams.beta_2), "weight_decay": 0},
        ])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
