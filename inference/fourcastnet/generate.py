#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FourCastNetV2 GRIB -> Numpy (.npy) with *step* dimension preserved.

- Keeps forecast lead dimension ("step"), exporting an array of shape (T, C, H, W)
- Cleanly separates isobaric levels from heightAboveGround groups to avoid cfgrib conflicts
- Supports choosing target isobaric levels via --levels_hpa (default 9-layer S2S set)
- Optional 6x6 spatial coarsening + latitude interpolation to 121 points (default on)
- Can include 2m temperature (t2m) via --include_t2m
- Prefers specific humidity (q) over relative humidity (r) if available

Typical usage (9 isobaric levels matching CirT S2S):
python generate_step_preserved.py \
  --input /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/fourcastnetv2-small.grib \
  --out_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet \
  --idx_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/.cfgrib_index \
  --levels_hpa 1000,925,850,700,600,500,400,300,250

Use all 13 available levels if desired (ensure downstream config matches exactly):
  --levels_hpa 50,100,150,200,250,300,400,500,600,700,850,925,1000
"""

import os
import argparse
import json
from typing import Callable, Dict, Optional, List

import numpy as np
import xarray as xr

import warnings
warnings.filterwarnings("ignore")

# Preferred order for pressure-level variables (maps to CirT var order)
# We'll try to use 'q' (specific humidity) if available; otherwise fallback to 'r'.
PL_VAR_ORDER_ABBR = ["z", "q_or_r", "t", "u", "v"]
SFC_VAR_ORDER_ABBR = ["u10", "v10"]

# Default 9-layer S2S set; override via --levels_hpa if needed
DEFAULT_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250]


# ----------------------- Helpers -----------------------
def _unique_idx_path(idx_dir: Optional[str], grib_path: str, fkeys: Dict[str, object]) -> Optional[str]:
    """Create a unique cfgrib index path per filter_by_keys set to avoid collisions."""
    if not idx_dir:
        return None
    os.makedirs(idx_dir, exist_ok=True)
    base = os.path.basename(grib_path)
    parts = [base] + [f"{k}_{v}" for k, v in sorted(fkeys.items())]
    return os.path.join(idx_dir, ".".join(map(str, parts)) + ".idx")


def open_cfgrib_filtered(path: str, idx_dir: Optional[str] = None, **filter_by_keys) -> xr.Dataset:
    backend_kwargs = {}
    idx_path = _unique_idx_path(idx_dir, path, filter_by_keys)
    if idx_path:
        backend_kwargs["indexpath"] = idx_path
    return xr.open_dataset(path, engine="cfgrib", filter_by_keys=filter_by_keys, backend_kwargs=backend_kwargs)


def drop_extra_dims_keep_step(da: xr.DataArray) -> xr.DataArray:
    """Drop nuisance dims but **keep** forecast step."""
    for d in ["number", "time", "valid_time"]:
        if d in da.dims:
            da = da.isel({d: 0})
    return da


def ensure_4d(arr: np.ndarray) -> np.ndarray:
    """Ensure shape is (T, 1, H, W) when given (T, H, W)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,H,W), got {arr.shape}")
    return arr[:, None, :, :]


def default_coarsen_and_interp(ds: xr.Dataset, lat_step: int = 6, lon_step: int = 6, out_lat: int = 121) -> xr.Dataset:
    """6x6 block mean + latitude interpolation to out_lat grid points."""
    ds_c = ds.coarsen(latitude=lat_step, longitude=lon_step, boundary="trim").mean()
    lat = ds_c.latitude
    lat_target = np.linspace(lat.values.max(), lat.values.min(), out_lat)
    return ds_c.interp(latitude=lat_target)


# ----------------------- Core logic -----------------------
def _resolve_pl_vars(ds_pl: xr.Dataset) -> List[str]:
    """Resolve pressure-level variables in desired order, preferring q over r."""
    have_q = "q" in ds_pl.data_vars
    have_r = "r" in ds_pl.data_vars
    pl_vars: List[str] = []
    for key in PL_VAR_ORDER_ABBR:
        if key == "q_or_r":
            if have_q:
                pl_vars.append("q")
            elif have_r:
                pl_vars.append("r")
            else:
                raise KeyError("Neither 'q' nor 'r' found in isobaricInhPa dataset.")
        else:
            if key not in ds_pl.data_vars:
                raise KeyError(f"Variable '{key}' not found in isobaricInhPa dataset.")
            pl_vars.append(key)
    return pl_vars


def build_tensor_from_grib(
    path: str,
    idx_dir: Optional[str],
    target_levels_hpa: List[int],
    coarsen_and_interp: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
    include_t2m: bool = False,
) -> (np.ndarray, Dict):
    """
    Read a GRIB file and return (tensor, meta):
      - tensor: (T, C, H, W)
      - meta:   dict with keys: levels_hpa, channel_order (abbr), used_q (bool)
    """
    # 1) Pressure levels
    ds_pl = open_cfgrib_filtered(path, idx_dir, typeOfLevel="isobaricInhPa")
    if coarsen_and_interp is not None:
        ds_pl = coarsen_and_interp(ds_pl)

    # decide variable list and whether q was used
    pl_vars = _resolve_pl_vars(ds_pl)  # e.g., ['z','q','t','u','v'] or with 'r'
    used_q = (pl_vars[1] == "q")

    channels = []
    channel_order = []  # human-readable order for logging/metadata

    for v in pl_vars:
        da = drop_extra_dims_keep_step(ds_pl[v])
        if "isobaricInhPa" not in da.dims:
            raise ValueError(f"Variable '{v}' lacks isobaricInhPa dimension.")
        da = da.sel(isobaricInhPa=target_levels_hpa, method="nearest")
        da = da.transpose("step", "isobaricInhPa", "latitude", "longitude")
        channels.append(da.values)  # (T, L, H, W)
        channel_order.extend([f"{v}@{lvl}hPa" for lvl in da.isobaricInhPa.values.tolist()])

    # 2) 10 m wind (u10, v10)
    ds_hag10 = open_cfgrib_filtered(path, idx_dir, typeOfLevel="heightAboveGround", level=10)
    if coarsen_and_interp is not None:
        ds_hag10 = coarsen_and_interp(ds_hag10)

    for v in SFC_VAR_ORDER_ABBR:
        if v not in ds_hag10:
            raise KeyError(f"Surface variable '{v}' not found in HAG=10 dataset.")
        da = drop_extra_dims_keep_step(ds_hag10[v]).transpose("step", "latitude", "longitude")
        channels.append(ensure_4d(da.values))  # (T, 1, H, W)
        channel_order.append(v)

    # 3) Optional: 2 m temperature
    if include_t2m:
        try:
            ds_hag2 = open_cfgrib_filtered(path, idx_dir, typeOfLevel="heightAboveGround", level=2)
            if coarsen_and_interp is not None:
                ds_hag2 = coarsen_and_interp(ds_hag2)
            if "t2m" not in ds_hag2:
                raise KeyError("'t2m' not found in HAG=2 dataset.")
            da = drop_extra_dims_keep_step(ds_hag2["t2m"]).transpose("step", "latitude", "longitude")
            channels.append(ensure_4d(da.values))
            channel_order.append("t2m")
        except Exception as e:
            raise RuntimeError(f"Failed to extract t2m: {e}")

    # Concatenate along channel axis -> (T, C, H, W)
    tensor = np.concatenate(channels, axis=1).astype(np.float32, copy=False)

    meta = {
        "levels_hpa": list(map(int, target_levels_hpa)),
        "channel_order": channel_order,
        "used_q": bool(used_q),
        "shape": list(map(int, tensor.shape)),
    }
    return tensor, meta


# ----------------------- CLI -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Convert FourCastNetV2 GRIB to .npy (T,C,H,W), keeping forecast steps")
    ap.add_argument("--input", required=True, help="Input GRIB file path")
    ap.add_argument("--out_dir", required=True, help="Output directory for .npy and .json")
    ap.add_argument("--idx_dir", default=None, help="Directory for cfgrib index files (optional but recommended)")
    ap.add_argument("--no_coarsen", action="store_true", help="Disable 6x6 coarsening & latitude interpolation")
    ap.add_argument("--lat_step", type=int, default=6, help="Latitude coarsen step")
    ap.add_argument("--lon_step", type=int, default=6, help="Longitude coarsen step")
    ap.add_argument("--out_lat", type=int, default=121, help="Latitude points after interpolation")
    ap.add_argument(
        "--levels_hpa",
        type=str,
        default=",".join(map(str, DEFAULT_LEVELS_HPA)),
        help="Comma-separated isobaric levels in hPa (order matters)",
    )
    ap.add_argument("--include_t2m", action="store_true", help="Also extract t2m (HAG=2m)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build the levels list
    levels_hpa = [int(x) for x in args.levels_hpa.split(",")] if args.levels_hpa else DEFAULT_LEVELS_HPA

    # Optional coarsening/interp function
    if args.no_coarsen:
        coarsen_fn = None
    else:
        def coarsen_fn(ds: xr.Dataset) -> xr.Dataset:
            return default_coarsen_and_interp(ds, lat_step=args.lat_step, lon_step=args.lon_step, out_lat=args.out_lat)

    tensor, meta = build_tensor_from_grib(
        path=args.input,
        idx_dir=args.idx_dir,
        target_levels_hpa=levels_hpa,
        coarsen_and_interp=coarsen_fn,
        include_t2m=args.include_t2m,
    )

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_npy = os.path.join(args.out_dir, f"{base}.npy")
    out_meta = os.path.join(args.out_dir, f"{base}.json")

    np.save(out_npy, tensor)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved array: {out_npy}  shape={tensor.shape}")
    print(f"📝 Saved meta : {out_meta}")
    print("Channel order (abbr):", ", ".join(meta["channel_order"]))


if __name__ == "__main__":
    main()
