#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- Phase 1: parse thread args EARLY (before importing numpy/xarray) ----------
import os, sys, argparse

def early_parse_and_set_env():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--omp-threads", type=int, default=4)
    p.add_argument("--mkl-threads", type=int, default=4)
    p.add_argument("--openblas-threads", type=int, default=4)
    p.add_argument("--numexpr-threads", type=int, default=4)
    # we only parse known thread args; leave others for the main parser
    args, _ = p.parse_known_args(sys.argv[1:])
    os.environ.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.mkl_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.openblas_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.numexpr_threads))

early_parse_and_set_env()

# ---------- Phase 2: heavy imports ----------
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import numpy as np
import pandas as pd
import xarray as xr

import config

import warnings
warnings.filterwarnings("ignore", message="Engine 'cfgrib' loading failed")

import zarr, numcodecs
def _ver_tuple(v):
    try: return tuple(int(p) for p in v.split(".")[:3])
    except Exception: return (0, 0, 0)
assert _ver_tuple(zarr.__version__) < (3, 0, 0), \
    f"Detected zarr {zarr.__version__}; please install zarr<3 (e.g., 2.16.1)"
assert (0,11,0) <= _ver_tuple(numcodecs.__version__) <= (0,12,99), \
    f"Detected numcodecs {numcodecs.__version__}; please use 0.11–0.12.x (e.g., 0.12.1)"

GLOBAL_DS = None

DEFAULT_URL = (
    "gs://weatherbench2/datasets/era5_daily/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr"
)

def process_one_day(
    date: pd.Timestamp,
    out_dir: str,
    single_level_vars,
    downsample_factor: int,
    retries: int,
    overwrite: bool,
    dask_scheduler: str,
    inner_dask_threads: int,
):
    from dask import config as dask_config
    if dask_scheduler == "threads":
        if inner_dask_threads and inner_dask_threads > 0:
            from multiprocessing.pool import ThreadPool
            dask_config.set(scheduler="threads", pool=ThreadPool(inner_dask_threads))
        else:
            dask_config.set(scheduler="threads")
    else:
        dask_config.set(scheduler="synchronous")

    global GLOBAL_DS
    ds = GLOBAL_DS
    if ds is None:
        raise RuntimeError("GLOBAL_DS is not initialized")

    date_str = date.strftime("%Y-%m-%d")
    ymd = date.strftime("%Y%m%d")
    out_path = os.path.join(out_dir, f"era5_single_full_1.5deg_{ymd}.zarr")

    if os.path.exists(out_path) and overwrite:
        shutil.rmtree(out_path, ignore_errors=True)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            sub = ds[single_level_vars].sel(time=date_str)
            sub = sub.isel(
                latitude=slice(None, None, downsample_factor),
                longitude=slice(None, None, downsample_factor),
            )
            sub = sub.fillna(0)
            os.makedirs(out_dir, exist_ok=True)
            sub.to_zarr(out_path, mode="w", consolidated=True)
            print(f"[OK]   {date_str} -> {out_path}")
            return out_path
        except Exception as e:
            last_err = e
            print(f"[RETRY {attempt}/{retries}] {date_str} failed: {e}")
            sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to process {date_str}: {last_err}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel ERA5 single-level daily -> 1.5° with controllable low-level threads."
    )
    p.add_argument("--url", type=str, default=DEFAULT_URL, help="Zarr store URL (gs://...)")
    p.add_argument("--start", type=str, default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default="2018-12-31", help="End date (YYYY-MM-DD) inclusive")
    p.add_argument("--workers", type=int, default=8, help="Outer per-day concurrency (threads)")
    p.add_argument("--retries", type=int, default=3, help="Retries per day")
    p.add_argument("--downsample_factor", type=int, default=6, help="0.25°->1.5° step (6)")
    p.add_argument("--output_dir", type=str, default=None,
                   help='Output dir (default: f"{config.DATA_DIR}/single_level_1.5")')
    p.add_argument("--anon", action="store_true",
                   help="Use anonymous access for public GCS (weatherbench2)")
    p.add_argument("--no-overwrite", action="store_true",
                   help="If set, do not overwrite existing outputs")
    p.add_argument("--scheduler", choices=["sync", "threads"], default="sync",
                   help="Per-task dask scheduler: sync (safe) or threads (faster, but nested)")
    p.add_argument("--inner-dask-threads", type=int, default=4,
                   help="Threads used by dask 'threads' scheduler INSIDE each task (0=auto)")
    p.add_argument("--omp-threads", type=int, default=8)
    p.add_argument("--mkl-threads", type=int, default=8)
    p.add_argument("--openblas-threads", type=int, default=8)
    p.add_argument("--numexpr-threads", type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()

    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    dates = pd.date_range(start_date, end_date, freq="D")

    output_dir = args.output_dir or os.path.join(config.DATA_DIR, "single_level_1.5")
    os.makedirs(output_dir, exist_ok=True)

    single_level_vars = config.ERA5_SINGLE_LEVEL  # e.g. ['t2m','u10','v10','msl','tp','sp','d2m',...]

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
        single_level_vars=single_level_vars,
        retries=args.retries,
        downsample_factor=args.downsample_factor,
        overwrite=overwrite,
        dask_scheduler=("threads" if args.scheduler == "threads" else "sync"),
        inner_dask_threads=args.inner_dask_threads,
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

if __name__ == "__main__":
    main()
