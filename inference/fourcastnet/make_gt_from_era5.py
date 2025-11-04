#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build GT (T, V, H, W) from daily ERA5 Zarrs (each file = one timestamp).
- No reliance on in-file 'time' dimension.
- Dates are taken from filenames like ..._YYYYMMDD.zarr and stacked along time.
- Strictly no cross-year unless you explicitly pass --date_from/--date_to.

Output: (T, V, 121, 240)
V order:
  PL: [geopotential, HUMIDITY, temperature, u_component_of_wind, v_component_of_wind] expanded by levels (var-major)
  SFC: [10m_u_component_of_wind, 10m_v_component_of_wind] (+ optional 2m_temperature)
"""

import argparse, os, glob, re
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict

import numpy as np
import xarray as xr
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# -------- Aliases --------
VAR_ALIASES: Dict[str, List[str]] = {
    "geopotential": ["geopotential", "z"],
    "temperature": ["temperature", "t"],
    "u_component_of_wind": ["u_component_of_wind", "u"],
    "v_component_of_wind": ["v_component_of_wind", "v"],
    "relative_humidity": ["relative_humidity", "r"],
    "specific_humidity": ["specific_humidity", "q"],
    "10m_u_component_of_wind": ["10m_u_component_of_wind", "u10"],
    "10m_v_component_of_wind": ["10m_v_component_of_wind", "v10"],
    "2m_temperature": ["2m_temperature", "t2m"],
}
PL_VARS_CANON_BASE = [
    "geopotential",
    "<HUMIDITY>",  # replaced by relative_humidity or specific_humidity
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
SFC_VARS_CANON = ["10m_u_component_of_wind", "10m_v_component_of_wind"]

DEFAULT_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250]

# filenames like era5_pressure_full_1.5deg_20180101.zarr
DATE_RE = re.compile(r"_(\d{8})(?:\.zarr)?$")


# -------- Helpers --------
def _path_date(p: str) -> Optional[date]:
    m = DATE_RE.search(os.path.basename(p))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()

def _sorted_zarrs(dir_path: str) -> List[str]:
    paths = glob.glob(os.path.join(dir_path, "*.zarr"))
    paths = [p for p in paths if _path_date(p)]
    return sorted(paths, key=_path_date)

def _filter_by_year_or_range(paths: List[str], year: Optional[int], date_from: Optional[str],
                             date_to: Optional[str], start_time: str) -> List[str]:
    if not year and not date_from and not date_to:
        y = int(start_time[:4])
        return [p for p in paths if (_pd := _path_date(p)) and _pd.year == y]
    if year and not date_from and not date_to:
        return [p for p in paths if (_pd := _path_date(p)) and _pd.year == year]
    d0 = datetime.strptime(date_from, "%Y%m%d").date() if date_from else date.min
    d1 = datetime.strptime(date_to, "%Y%m%d").date() if date_to else date.max
    return [p for p in paths if (_pd := _path_date(p)) and d0 <= _pd <= d1]

def _open_zarr(path: str) -> xr.Dataset:
    # daily zarrs already small; consolidated metadata is recommended
    return xr.open_zarr(path, consolidated=True)

def _standardize_dims(ds: xr.Dataset, is_pl: bool) -> xr.Dataset:
    # lat/lon
    if "latitude" not in ds.dims and "lat" in ds.dims:
        ds = ds.rename({"lat": "latitude"})
    if "longitude" not in ds.dims and "lon" in ds.dims:
        ds = ds.rename({"lon": "longitude"})
    # pressure levels
    if is_pl:
        if "isobaricInhPa" not in ds.dims:
            if "level" in ds.dims:
                ds = ds.rename({"level": "isobaricInhPa"})  # your files show 'level'
            elif "isobaricInPa" in ds.dims:
                ds = ds.assign_coords(isobaricInhPa=(ds["isobaricInPa"] / 100.0))
                ds = ds.swap_dims({"isobaricInPa": "isobaricInhPa"}).drop_vars("isobaricInPa")
    return ds

def _resolve_and_rename_vars(ds: xr.Dataset, needed_canon: List[str]) -> Optional[xr.Dataset]:
    rename_map = {}
    ds_vars = set(ds.data_vars)
    for canon in needed_canon:
        found = None
        for cand in VAR_ALIASES.get(canon, [canon]):
            if cand in ds_vars:
                found = cand
                break
        if not found:
            return None
        if found != canon:
            rename_map[found] = canon
    if rename_map:
        ds = ds.rename(rename_map)
    return ds[needed_canon]

def _coarsen_and_interp(ds: xr.Dataset, lat_step=6, lon_step=6, out_lat=121) -> xr.Dataset:
    ds_c = ds.coarsen(latitude=lat_step, longitude=lon_step, boundary="trim").mean()
    lat = ds_c.latitude
    lat_target = np.linspace(lat.values.max(), lat.values.min(), out_lat)
    return ds_c.interp(latitude=lat_target)


# -------- Core loaders (file-by-file; no 'time' in file) --------
def _load_pl_one(path: str, pl_vars_canon: List[str], levels_hpa: List[int],
                 do_coarsen: bool, lat_step: int, lon_step: int, out_lat: int) -> np.ndarray:
    ds = _open_zarr(path)
    ds = _standardize_dims(ds, is_pl=True)
    ds = _resolve_and_rename_vars(ds, pl_vars_canon)
    if ds is None:
        raise KeyError(f"{path} missing required PL vars: {pl_vars_canon}")
    if "isobaricInhPa" not in ds.dims:
        raise ValueError(f"{path} missing isobaricInhPa dimension")

    # select levels; then optional coarsen/interp
    ds = ds.sel(isobaricInhPa=levels_hpa, method="nearest")
    if do_coarsen:
        ds = _coarsen_and_interp(ds, lat_step, lon_step, out_lat)

    # stack channels var-major across levels
    chans = []
    for v in pl_vars_canon:
        da = ds[v].transpose("isobaricInhPa", "latitude", "longitude")
        chans.append(da.values)  # (L,H,W)
    return np.concatenate(chans, axis=0)  # (V_pl, H, W)

def _load_sfc_one(path: str, sfc_vars_canon: List[str],
                  do_coarsen: bool, lat_step: int, lon_step: int, out_lat: int) -> np.ndarray:
    ds = _open_zarr(path)
    ds = _standardize_dims(ds, is_pl=False)
    ds = _resolve_and_rename_vars(ds, sfc_vars_canon)
    if ds is None:
        raise KeyError(f"{path} missing required SFC vars: {sfc_vars_canon}")

    if do_coarsen:
        ds = _coarsen_and_interp(ds, lat_step, lon_step, out_lat)

    chans = []
    for v in sfc_vars_canon:
        da = ds[v].transpose("latitude", "longitude")
        arr = da.values[None, :, :]  # (1,H,W)
        chans.append(arr)
    return np.concatenate(chans, axis=0)  # (V_sfc, H, W)


def build_gt_from_dirs(
    pl_dir: str,
    sfc_dir: str,
    start_time: str,
    steps: int,
    levels_hpa: List[int],
    humidity: str,
    no_coarsen: bool,
    lat_step: int,
    lon_step: int,
    out_lat: int,
    include_t2m: bool,
    year: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
) -> Tuple[np.ndarray, List[str]]:
    # humidity canonical
    if humidity not in {"relative", "specific"}:
        raise ValueError("--humidity must be 'relative' or 'specific'")
    hum_canon = "relative_humidity" if humidity == "relative" else "specific_humidity"

    # list & filter files
    pl_all = _sorted_zarrs(pl_dir)
    sfc_all = _sorted_zarrs(sfc_dir)
    if not pl_all:
        raise FileNotFoundError(f"No .zarr found under {pl_dir}")
    if not sfc_all:
        raise FileNotFoundError(f"No .zarr found under {sfc_dir}")

    pl_paths = _filter_by_year_or_range(pl_all, year, date_from, date_to, start_time)
    sfc_paths = _filter_by_year_or_range(sfc_all, year, date_from, date_to, start_time)
    print(f"PL candidates: {len(pl_paths)}   SFC candidates: {len(sfc_paths)}")

    # choose common dates >= start_date
    start_date = datetime.fromisoformat(start_time.replace("Z", "")).date()
    pl_map = { _path_date(p): p for p in pl_paths }
    sfc_map = { _path_date(p): p for p in sfc_paths }
    common_dates = sorted([d for d in pl_map.keys() if d in sfc_map and d is not None and d >= start_date])

    if not common_dates:
        raise ValueError("No common dates between PL and SFC after start_time in filtered files.")
    # pick first T dates (T=steps when each file is one timestamp)
    take_dates = common_dates[:steps]
    if len(take_dates) < steps:
        print(f"[warn] Only {len(take_dates)} dates available; requested {steps}.")

    # canonical var lists
    pl_vars_canon = [v if v != "<HUMIDITY>" else hum_canon for v in PL_VARS_CANON_BASE]
    sfc_vars_canon = list(SFC_VARS_CANON) + (["2m_temperature"] if include_t2m else [])

    # load date by date
    do_coarsen = (not no_coarsen)
    frames = []
    for d in tqdm(take_dates, desc="load daily GT", dynamic_ncols=True):
        pl_arr = _load_pl_one(pl_map[d], pl_vars_canon, levels_hpa, do_coarsen, lat_step, lon_step, out_lat)
        sfc_arr = _load_sfc_one(sfc_map[d], sfc_vars_canon, do_coarsen, lat_step, lon_step, out_lat)
        # concat channels
        vh = np.concatenate([pl_arr, sfc_arr], axis=0)  # (V,H,W)
        frames.append(vh)
    gt = np.stack(frames, axis=0).astype(np.float32, copy=False)  # (T,V,H,W)

    # build channel order strings
    channel_order = []
    # need actual selected levels (after nearest)
    # reopen first PL to read nearest-mapped levels
    ds0 = _open_zarr(pl_map[take_dates[0]])
    ds0 = _standardize_dims(ds0, is_pl=True)
    ds0 = _resolve_and_rename_vars(ds0, pl_vars_canon)
    ds0 = ds0.sel(isobaricInhPa=levels_hpa, method="nearest")
    levels_used = [int(x) for x in ds0["isobaricInhPa"].values.tolist()]
    for v in pl_vars_canon:
        channel_order.extend([f"{v}@{lvl}hPa" for lvl in levels_used])
    for v in sfc_vars_canon:
        channel_order.append(v)

    return gt, channel_order


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Build GT (T,V,H,W) from daily ERA5 Zarrs (one timestamp per file)")
    ap.add_argument("--pl_dir", required=True)
    ap.add_argument("--sfc_dir", required=True)
    ap.add_argument("--start_time", default="2018-01-01T00:00:00")
    ap.add_argument("--steps", type=int, default=41)  # here: 41 days if one ts per file
    ap.add_argument("--levels_hpa", type=str, default=",".join(map(str, DEFAULT_LEVELS_HPA)))
    ap.add_argument("--humidity", choices=["relative", "specific"], default="specific",
                    help="Your PL zarr shows 'specific_humidity', so default here is 'specific'")
    ap.add_argument("--include_t2m", action="store_true")
    ap.add_argument("--no_coarsen", action="store_true")
    ap.add_argument("--lat_step", type=int, default=6)
    ap.add_argument("--lon_step", type=int, default=6)
    ap.add_argument("--out_lat", type=int, default=121)
    ap.add_argument("--year", type=int, help="Only scan this year (no cross-year)")
    ap.add_argument("--date_from", type=str, help="YYYYMMDD")
    ap.add_argument("--date_to", type=str, help="YYYYMMDD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels_hpa.split(",")]
    gt, order = build_gt_from_dirs(
        pl_dir=args.pl_dir,
        sfc_dir=args.sfc_dir,
        start_time=args.start_time,
        steps=args.steps,
        levels_hpa=levels,
        humidity=args.humidity,
        no_coarsen=args.no_coarsen,
        lat_step=args.lat_step,
        lon_step=args.lon_step,
        out_lat=args.out_lat,
        include_t2m=args.include_t2m,
        year=args.year,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, gt)
    print("✅ GT saved:", args.out, "shape=", gt.shape)
    print("channel_order:", ", ".join(order))


if __name__ == "__main__":
    main()
