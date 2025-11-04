#!/usr/bin/env python3
"""
Evaluate FourCastNetV2 predictions against ERA5 ground truth and export a per-variable CSV.

Key features (v2):
- Accepts prediction shapes (V,H,W), (T,V,H,W) or (S,T,V,H,W); treats missing dims as S=1 and/or T=1.
- Allows **--skip_denorm** since FourCastNet outputs are already in physical units.
- Works with CirT metrics (RMSE, Bias, ACC, MS-SSIM, SpectralDiv/Res).
- Loads GT either from a provided --gt_npy or from CirT's test dataloader; clips to common S,T if lengths differ.
- Assumes channel order = pred_pressure_vars expanded over CIRT.config.PRESSURE_LEVELS, then pred_single_vars.

Typical usage (no denorm; saves CSV in fourcastnet folder):
python evaluate_fourcastnet.py \
  --pred_npy /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/fourcastnetv2-small.npy \
  --config_filepath fourcastnet_infer.yaml \
  --output_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet \
  --save_name fourcastnetv2_metrics.csv \
  --skip_denorm
"""

import argparse
import os
import warnings
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import torch
import xarray as xr
import lightning.pytorch as pl
from tqdm import tqdm

from CIRT import criterion, config

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "disabled"
pl.seed_everything(42)

test_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ------------------------------
# Normalization utilities
# ------------------------------

def _open_norm_ds_pressure():
    return xr.open_dataset(
        Path("/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT_old/data/S2S")
        / "climatology_1.5"
        / "climatology_pressure_level_1.5_new.zarr",
        engine="zarr",
    )


def _open_norm_ds_single():
    return xr.open_dataset(
        Path("/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT_old/data/S2S")
        / "climatology_1.5"
        / "climatology_single_level_1.5_new.zarr",
        engine="zarr",
    )


def reverse_normalize(t: torch.Tensor, data_args: dict) -> torch.Tensor:
    """Reverse standardization using climatology mean/sigma (expects CirT var names).
    Only use if your predictions/GT are standardized. FourCastNet predictions are
    typically *already in physical units*; prefer --skip_denorm.
    """
    assert t.ndim == 5, f"expect [S,T,V,H,W], got {t.shape}"
    device = t.device

    pred_single_vars = list(data_args.get("pred_single_vars", []))
    pred_pressure_vars = list(data_args.get("pred_pressure_vars", []))

    # Guard: if using relative_humidity (r) but your climatology is for specific_humidity (q),
    # denorm will be incorrect or fail. Recommend --skip_denorm in that case.
    if any(v == "relative_humidity" for v in pred_pressure_vars):
        raise RuntimeError(
            "reverse_normalize: 'relative_humidity' detected. Unless you have RH climatology, "
            "please rerun with --skip_denorm."
        )

    ds_p = _open_norm_ds_pressure()
    ds_s = _open_norm_ds_single()

    pressure_params = [f"{p}-{lvl}" for p in pred_pressure_vars for lvl in config.PRESSURE_LEVELS]

    mean_p = torch.tensor(ds_p["mean"].sel(param=pressure_params).values[:, None, None], dtype=torch.float32)
    std_p = torch.tensor(ds_p["sigma"].sel(param=pressure_params).values[:, None, None], dtype=torch.float32)

    mean_s = torch.tensor(ds_s["mean"].sel(param=pred_single_vars).values[:, None, None], dtype=torch.float32)
    std_s = torch.tensor(ds_s["sigma"].sel(param=pred_single_vars).values[:, None, None], dtype=torch.float32)

    mean = torch.cat([mean_p, mean_s], dim=0).to(device)
    std = torch.cat([std_p, std_s], dim=0).to(device)

    return t * std + mean


# ------------------------------
# Features & Metrics
# ------------------------------

def build_feature_names(data_args: dict):
    names = []
    for p in data_args["pred_pressure_vars"]:
        for lvl in config.PRESSURE_LEVELS:
            names.append(f"{p}-{lvl}")
    names += list(data_args["pred_single_vars"])  # maintain order
    return names


def calculate_metrics(all_pred: torch.Tensor, all_y: torch.Tensor, data_args: dict) -> pd.DataFrame:
    """Compute metrics per (step, variable) with CirT's criteria. Shapes: [S,T,V,H,W]."""
    if isinstance(all_pred, np.ndarray):
        all_pred = torch.from_numpy(all_pred)
    if isinstance(all_y, np.ndarray):
        all_y = torch.from_numpy(all_y)

    device = torch.device("cpu")
    all_pred = all_pred.to(device)
    all_y = all_y.to(device)

    feature_names = build_feature_names(data_args)

    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    ACC = criterion.ACC()
    MS_SSIM = criterion.MS_SSIM()
    SpecDiv = criterion.SpectralDiv(percentile=0.9, is_train=False)
    SpecRes = criterion.SpectralRes(percentile=0.9, is_train=False)

    cols = ["steps", "pred_vars", "RMSE", "Bias", "ACC", "MS_SSIM", "SpecDiv", "SpecRes"]
    df = pd.DataFrame(columns=cols)

    S, T, V, H, W = all_pred.shape
    print(f"Evaluating: S={S}, T={T}, V={V}, H={H}, W={W}")

    if V != len(feature_names):
        msg = (
            f"Variable/channel mismatch: pred V={V} vs expected {len(feature_names)}.\n"
            f"- Ensure --levels_hpa in exporter exactly matches CIRT.config.PRESSURE_LEVELS.\n"
            f"- Ensure pred_pressure_vars / pred_single_vars match your exporter order.\n"
            f"- pred_pressure_vars={data_args['pred_pressure_vars']}, pred_single_vars={data_args['pred_single_vars']}\n"
        )
        raise ValueError(msg)

    for t_idx in range(T):
        yhat_t = all_pred[:, t_idx]
        y_t = all_y[:, t_idx]

        for i, name in enumerate(tqdm(feature_names, desc=f"Step {t_idx+1}/{T}")):
            src = "pressure_level" if "-" in name else "single_level"
            yhat = yhat_t[:, i]
            y = y_t[:, i]

            rmse = RMSE(yhat, y)
            bias = Bias(yhat, y)
            acc = ACC(yhat, y, name, src)
            ms = MS_SSIM(yhat, y)
            sd = SpecDiv(yhat, y)
            sr = SpecRes(yhat, y)

            df.loc[len(df)] = [t_idx, name, f"{rmse:.4f}", f"{bias:.4f}", f"{acc:.4f}", f"{ms:.4f}", f"{sd:.4f}", f"{sr:.4f}"]

    return df


# ------------------------------
# GT loading (optional from dataloader)
# ------------------------------

def _load_gt_from_dataloader(model_args: dict, data_args: dict) -> torch.Tensor:
    """Build CirT test dataloader and stack ground truth Y to [S,T,V,H,W]."""
    try:
        from CIRT.models import model as cirt_model
        print("Building test dataloader from CirT config to fetch GT...")
        m = cirt_model.S2SBenchmarkModel(model_args=model_args, data_args=data_args)
        m.setup()
        dl = m.test_dataloader()

        ys = []
        for batch in dl:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    _, y, _ = batch
                elif len(batch) == 2:
                    _, y = batch
                else:
                    raise ValueError("Unexpected test batch format from dataloader.")
            elif isinstance(batch, dict):
                y = batch.get("y") or batch.get("target")
                if y is None:
                    raise ValueError("Dict batch missing 'y' or 'target'.")
            else:
                raise ValueError("Unsupported batch type from dataloader.")
            ys.append(y)

        all_y = torch.cat(ys, dim=0)
        print(f"GT stacked from dataloader: {tuple(all_y.shape)}")
        return all_y
    except Exception as e:
        raise RuntimeError(
            "Failed to build GT from dataloader. Provide --gt_npy explicitly.\n"
            f"Root cause: {e}"
        )


# ------------------------------
# Shape utilities
# ------------------------------

def _to_STVHW(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to shape [S,T,V,H,W]."""
    if arr.ndim == 3:        # (V,H,W)
        arr = arr[None, None, ...]
    elif arr.ndim == 4:      # (T,V,H,W)
        arr = arr[None, ...]
    elif arr.ndim == 5:      # (S,T,V,H,W)
        pass
    else:
        raise ValueError(f"Unsupported array shape {arr.shape}; expected (V,H,W)/(T,V,H,W)/(S,T,V,H,W)")
    return arr


# ------------------------------
# Main
# ------------------------------

def main(args):
    # Load config for variables & dataset
    with open(args.config_filepath, "r") as f:
        hyper = yaml.load(f, Loader=yaml.FullLoader)
    model_args = hyper["model_args"]
    data_args = hyper["data_args"]

    # Load predictions
    pred_path = Path(args.pred_npy)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    print(f"Loading predictions: {pred_path}")
    pred_np = np.load(pred_path)
    all_pred = _to_STVHW(pred_np)

    # Load GT
    if args.gt_npy:
        gt_path = Path(args.gt_npy)
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        print(f"Loading ground truth: {gt_path}")
        gt_np = np.load(gt_path)
        all_y = _to_STVHW(gt_np)
    else:
        all_y = _to_STVHW(_load_gt_from_dataloader(model_args, data_args).numpy())

    # Align shapes on S,T and check V/H/W
    if all_pred.shape[2:] != all_y.shape[2:]:
        raise ValueError(
            f"Spatial/variable mismatch: pred {all_pred.shape[2:]} vs GT {all_y.shape[2:]}"
        )
    S = min(all_pred.shape[0], all_y.shape[0])
    T = min(all_pred.shape[1], all_y.shape[1])

    all_pred = torch.from_numpy(all_pred[:S, :T])
    all_y = torch.from_numpy(all_y[:S, :T])

    print(f"After align: pred {tuple(all_pred.shape)}, gt {tuple(all_y.shape)}")

    # Optional reverse normalization (not recommended for FourCastNet outputs)
    if not args.skip_denorm:
        print("Reversing normalization...")
        all_pred = reverse_normalize(all_pred, data_args)
        all_y = reverse_normalize(all_y, data_args)
    else:
        print("Skipping reverse normalization (--skip_denorm)")

    # Metrics
    print("Calculating metrics...")
    metrics_df = calculate_metrics(all_pred, all_y, data_args)

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.save_name or f"fourcastnetv2_metrics_{test_time}.csv"
    csv_path = out_dir / base
    metrics_df.to_csv(csv_path, index=False)

    # Summary
    print(f"\nSaved metrics to: {csv_path}")
    for step in metrics_df["steps"].unique():
        step_df = metrics_df[metrics_df["steps"] == step]
        print(
            f"Step {int(step)+1}: RMSE={step_df['RMSE'].astype(float).mean():.4f}, "
            f"ACC={step_df['ACC'].astype(float).mean():.4f}, "
            f"MS_SSIM={step_df['MS_SSIM'].astype(float).mean():.4f}"
        )

    return str(csv_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate FourCastNetV2 predictions and export metrics CSV")
    p.add_argument("--pred_npy", required=True, help="Path to predictions .npy; accepts (VHW)/(TVHW)/(STVHW)")
    p.add_argument("--gt_npy", default=None, help="Optional path to ground truth .npy; accepts (VHW)/(TVHW)/(STVHW)")
    p.add_argument("--config_filepath", default="CIRT/configs/CirT.yaml", help="CirT YAML config path")
    p.add_argument(
        "--output_dir",
        default="/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet",
        help="Directory to save the CSV",
    )
    p.add_argument("--save_name", default=None, help="Optional CSV filename (default uses timestamp)")
    p.add_argument("--skip_denorm", action="store_true", help="Pred/GT already in physical units; skip reverse normalization")

    args = p.parse_args()

    try:
        out = main(args)
        print(f"\n✅ Done. CSV: {out}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
