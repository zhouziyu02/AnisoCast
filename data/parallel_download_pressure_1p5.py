#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import numpy as np
import pandas as pd
import xarray as xr

import config
from download_utils import parse_date, positive_int, skip_existing_store, validate_range

import warnings
warnings.filterwarnings("ignore", message="Engine 'cfgrib' loading failed")

GLOBAL_DS = None

DEFAULT_URL = (
    "gs://weatherbench2/datasets/era5_daily/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr"
)


def process_one_day(
    date: pd.Timestamp,
    out_dir: str,
    era5_pressure_vars,
    pressure_levels,
    downsample_factor: int = 6,
    retries: int = 3,
    overwrite: bool = True,
):
    global GLOBAL_DS
    ds = GLOBAL_DS
    if ds is None:
        raise RuntimeError("GLOBAL_DS is not initialized")

    date_str = date.strftime("%Y-%m-%d")
    ymd = date.strftime("%Y%m%d")
    out_path = os.path.join(out_dir, f"era5_pressure_full_1.5deg_{ymd}.zarr")

    if skip_existing_store(out_path, era5_pressure_vars, date, overwrite):
        return out_path

    if os.path.exists(out_path) and overwrite:
        shutil.rmtree(out_path, ignore_errors=True)

    last_err = None
    for attempt in range(1, retries + 1):
        owns_output = overwrite
        try:
            sub = ds.sel(time=slice(date_str, date_str))[era5_pressure_vars]
            if sub.sizes.get("time", 0) == 0:
                raise ValueError(f"No source samples for {date_str}")
            sub = sub.sel(level=pressure_levels)

            # 0.25° -> 1.5°
            sub = sub.isel(
                latitude=slice(None, None, downsample_factor),
                longitude=slice(None, None, downsample_factor),
            )

            sub = sub.fillna(0)
            sub.attrs.update(missing_values="filled with zero", spatial_sampling="index stride", downsample_factor=downsample_factor)
            for variable in sub.variables:
                sub[variable].encoding = {}
            os.makedirs(out_dir, exist_ok=True)
            if not overwrite:
                os.mkdir(out_path)  # Exclusive reservation prevents concurrent overwrites.
                owns_output = True
            pending = sub.to_zarr(out_path, mode="w", consolidated=True, compute=False)
            pending.compute(scheduler="synchronous")
            print(f"[OK]   {date_str} -> {out_path}")
            return out_path

        except Exception as e:
            last_err = e
            if owns_output:
                shutil.rmtree(out_path, ignore_errors=True)
            print(f"[RETRY {attempt}/{retries}] {date_str} failed: {e}")
            if attempt < retries:
                sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to process {date_str}: {last_err}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Low-memory parallel download & downsample ERA5 pressure-level daily data to 1.5°."
    )
    p.add_argument("--url", type=str, default=DEFAULT_URL,
                   help="Zarr store URL on GCS (gs://...)")
    p.add_argument("--start", type=parse_date, default=parse_date("1979-01-01"),
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=parse_date, default=parse_date("2018-12-31"),
                   help="End date (YYYY-MM-DD) (inclusive)")
    p.add_argument("--workers", type=positive_int, default=4,
                   help="Max concurrent workers (2–6 recommended; adjust for available memory)")
    p.add_argument("--retries", type=positive_int, default=3,
                   help="Retries per day")
    p.add_argument("--downsample_factor", type=positive_int, default=6,
                   help="Sampling stride from 0.25° to 1.5° (default: 6)")
    p.add_argument("--output_dir", type=str, default=None,
                   help='Output dir (default: f"{config.DATA_DIR}/pressure_level_1.5")')
    p.add_argument("--anon", action="store_true",
                   help="Use anonymous access for public GCS (recommended for weatherbench2)")
    p.add_argument("--no-overwrite", action="store_true",
                   help="Do not overwrite existing outputs (default: overwrite)")
    return p.parse_args()


def main():
    import fsspec

    args = parse_args()

    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    validate_range(start_date, end_date)
    dates = pd.date_range(start_date, end_date, freq="D")

    output_dir = args.output_dir or os.path.join(config.DATA_DIR, "pressure_level_1.5")
    os.makedirs(output_dir, exist_ok=True)

    era5_pressure_vars = config.ERA5_PRESSURE_LEVEL
    pressure_levels    = config.PRESSURE_LEVELS

    storage_options = {"token": "anon"} if args.anon else None

    global GLOBAL_DS
    GLOBAL_DS = xr.open_zarr(
        args.url,
        consolidated=True,
        chunks="auto",
        storage_options=storage_options,
    )


    overwrite = not args.no_overwrite

    worker_kwargs = dict(
        out_dir=output_dir,
        era5_pressure_vars=era5_pressure_vars,
        pressure_levels=pressure_levels,
        retries=args.retries,
        downsample_factor=args.downsample_factor,
        overwrite=overwrite,
    )

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_day, d, **worker_kwargs) for d in dates]
        for fut in as_completed(futures):
            try:
                fut.result()
                ok += 1
            except Exception as e:
                print("[FAIL]", e)
                fail += 1

    try:
        GLOBAL_DS.close()
    except Exception:
        pass

    print(f"\nDone. Success: {ok}, Failed: {fail}, Output dir: {output_dir}")
    return 1 if fail else 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    raise SystemExit(main())
