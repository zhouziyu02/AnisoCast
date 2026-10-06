#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast climatology over (time, lat/lon) for ERA5 WB2 1.5°

Features
- Parallel & distributed with Dask (LocalCluster)
- Parallel Zarr open (thread-pool): --open_concurrency (preserves file order)
- Two aggregation modes:
    * --agg concat   : concat along time -> mean/std over (time,lat,lon)
    * --agg pairwise : (default) per-shard count/mean/centered-M2 over time and space,
                       then stable moment merging -> mean/std (ddof=0), same result as above,
                       but with much smaller task graphs and better throughput
- Raw Zarr fast path with explicit dimension metadata and CF decoding: --force_raw_zarr
- Training period defaults to 1979-01-01 through 2016-12-31; every requested day is required
- Progress bars: --progress none | vars | vars+files
- Dimension name adaptation: lat<->latitude, lon<->longitude
- Force overwrite outputs (never skip if exists)

Output path:
  {config.DATA_DIR}/climatology_1.5/climatology_{dataset_name}{out_suffix}.zarr

Example (pressure levels, many shards):
  python calculate_climatology.py \
    --dataset_name pressure_level_1.5 \
    --agg pairwise --force_raw_zarr --progress vars+files \
    --scheduler distributed --n_workers 16 --threads_per_worker 1 --memory_limit 6GB \
    --open_concurrency 32 \
    --chunks "time:64,latitude:180,longitude:180"
"""

import argparse
import logging
import warnings
import shutil
import os
import re

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

import zarr
import dask.array as da

from download_utils import parse_date, validate_range

import config  # must define: DATA_DIR, ERA5_SINGLE_LEVEL, ERA5_PRESSURE_LEVEL, PRESSURE_LEVELS (for PL reorder)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ----------------- helpers -----------------

def parse_chunks(spec: str):
    ck = {}
    if not spec:
        return ck
    for kv in spec.split(","):
        k, v = kv.split(":")
        ck[k.strip()] = int(v)
    return ck


def adapt_chunks_to_da(chunk_dict, da_x: xr.DataArray):
    if not chunk_dict or da_x is None:
        return {}
    aliases = {
        "lat": ("lat", "latitude"),
        "latitude": ("lat", "latitude"),
        "lon": ("lon", "longitude"),
        "longitude": ("lon", "longitude"),
        "time": ("time",),
    }
    present = set(da_x.dims)
    out = {}
    for k, v in chunk_dict.items():
        if k in present:
            out[k] = v
            continue
        cands = aliases.get(k, (k,))
        placed = False
        for cand in cands:
            if cand in present:
                out[cand] = v
                placed = True
                break
        if not placed:
            for _, cands in aliases.items():
                if k in cands:
                    for cand in cands:
                        if cand in present:
                            out[cand] = v
                            placed = True
                            break
                    if placed:
                        break
    return out


def open_zarr_flexible(path: Path, consolidated=True) -> xr.Dataset:
    store = str(path)
    if consolidated:
        try:
            return xr.open_zarr(store, consolidated=True)
        except Exception as e:
            logging.warning(f"open_zarr(consolidated=True) failed for {path}: {e}; retry without consolidated.")
    return xr.open_zarr(store)


def dims_to_reduce(da_x: xr.DataArray):
    reduce = []
    for d in ("time", "lat", "latitude", "lon", "longitude"):
        if d in da_x.dims:
            reduce.append(d)
    return tuple(reduce)


# ---------- Fallback: build DataArray from raw Zarr ---------

def open_var_via_zarr(path: Path, var: str) -> xr.DataArray:
    """Read explicit dimension metadata and decode CF scaling, fill values and time."""
    group = zarr.open_group(str(path), mode="r")
    if var not in group:
        raise KeyError(f"{path}: required variable {var!r} is missing")

    def read_array(name, lazy=False):
        array = group[name]
        attrs = dict(array.attrs)
        dims = attrs.pop("_ARRAY_DIMENSIONS", None)
        if dims is None or len(dims) != array.ndim:
            raise ValueError(f"{path}/{name}: missing or invalid _ARRAY_DIMENSIONS")
        if array.fill_value is not None:
            attrs.setdefault("_FillValue", array.fill_value)
        values = da.from_zarr(array) if lazy else np.asarray(array[()] if array.ndim == 0 else array[:])
        return xr.DataArray(values, dims=dims, attrs=attrs, name=name)

    array = read_array(var, lazy=True)
    coords = {}
    for name in ("time", "level", "latitude", "longitude", "lat", "lon"):
        if name in group:
            coordinate = read_array(name)
            if set(coordinate.dims).issubset(array.dims):
                coords[name] = coordinate
    return xr.decode_cf(xr.Dataset({var: array}, coords=coords))[var]


def _open_one(fp, var, chunk_dict, consolidated, force_raw_zarr, start, end):
    try:
        if force_raw_zarr:
            array = open_var_via_zarr(fp, var)
        else:
            dataset = open_zarr_flexible(fp, consolidated)
            if var not in dataset:
                raise KeyError(f"Required variable {var!r} is missing")
            array = dataset[var]
        if "time" not in array.coords:
            raise ValueError("A time coordinate is required")
        if "time" not in array.dims:
            array = array.expand_dims("time")
        if not np.issubdtype(array.time.dtype, np.datetime64):
            raise ValueError("Time must decode to calendar dates")
        array = array.sel(time=slice(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())))
        if array.sizes["time"] == 0:
            return None
        chunks = adapt_chunks_to_da(chunk_dict, array)
        return (array.chunk(chunks) if chunks else array).astype("float64")
    except Exception as exc:
        raise ValueError(f"Cannot read {var!r} from {fp}: {exc}") from exc


def parallel_open_list(files, var, chunk_dict, consolidated, force_raw_zarr,
                       show_file_tqdm=False, open_concurrency=1,
                       start="1979-01-01", end="2016-12-31"):
    """Open shards in order, enforcing complete dates and consistent coordinates."""
    def read(path):
        return _open_one(path, var, chunk_dict, consolidated, force_raw_zarr, start, end)
    with ThreadPoolExecutor(max_workers=open_concurrency) as executor:
        iterator = executor.map(read, files)
        if show_file_tqdm:
            iterator = tqdm(iterator, total=len(files), desc=f"{var}: open", leave=False)
        arrays = [array for array in iterator if array is not None]
    if not arrays:
        raise ValueError(f"No {var} samples in the requested period")
    reference = arrays[0]
    for array in arrays[1:]:
        if set(array.dims) != set(reference.dims):
            raise ValueError(f"{var}: inconsistent dimensions between shards")
        for dim in reference.dims:
            if dim != "time" and not reference[dim].equals(array[dim]):
                raise ValueError(f"{var}: inconsistent {dim} coordinates between shards")
    times = pd.DatetimeIndex(np.concatenate([array.time.values for array in arrays]))
    if times.has_duplicates:
        raise ValueError(f"{var}: duplicate timestamps across input shards")
    missing = pd.date_range(start, end, freq="D").difference(times.normalize().unique())
    if len(missing):
        examples = ", ".join(day.strftime("%Y-%m-%d") for day in missing[:5])
        raise ValueError(f"{var}: {len(missing)} requested day(s) are missing (e.g. {examples})")
    return arrays


# ---------- aggregation modes ----------

def concat_mode(dataset_files, var, chunk_dict, consolidated,
                show_file_tqdm, force_raw_zarr, files_limit, open_concurrency, start, end):
    files = dataset_files[:files_limit] if files_limit else dataset_files
    da_list = parallel_open_list(files, var, chunk_dict, consolidated, force_raw_zarr,
                                 show_file_tqdm=show_file_tqdm, open_concurrency=open_concurrency, start=start, end=end)
    if not da_list:
        return None, None
    da_all = xr.concat(da_list, dim="time", join="exact")  # lazy concat along time (chronological)
    reduce_dims = dims_to_reduce(da_all)
    if not reduce_dims:
        return None, None
    da_mean = da_all.mean(dim=reduce_dims, skipna=True)
    da_std = da_all.std(dim=reduce_dims, skipna=True)
    with ProgressBar():
        mean_v, std_v = dask.compute(da_mean, da_std)
    return mean_v, std_v


def reduce_moments(array):
    """Reduce all time and spatial samples with stable float64 centered moments."""
    array = array.astype("float64")
    reduce = dims_to_reduce(array)
    count = array.count(reduce)
    mean = array.mean(reduce, skipna=True).where(count > 0, 0)
    m2 = ((array - mean) ** 2).sum(reduce, skipna=True)
    return count, mean, m2


def pairwise_mode(dataset_files, var, chunk_dict, consolidated,
                  show_file_tqdm, force_raw_zarr, files_limit, open_concurrency, start, end):
    files = dataset_files[:files_limit] if files_limit else dataset_files
    arrays = parallel_open_list(files, var, chunk_dict, consolidated, force_raw_zarr,
                                show_file_tqdm, open_concurrency, start, end)
    moments = [reduce_moments(array) for array in arrays]
    # Chan's variance formula in a balanced tree keeps graph depth logarithmic.
    while len(moments) > 1:
        merged = []
        for index in range(0, len(moments), 2):
            if index + 1 == len(moments):
                merged.append(moments[index])
                continue
            n1, mean1, m21 = moments[index]
            n2, mean2, m22 = moments[index + 1]
            count = n1 + n2
            denominator = count.where(count > 0, 1)
            delta = mean2 - mean1
            merged.append((count, mean1 + delta * n2 / denominator,
                           m21 + m22 + delta ** 2 * n1 * n2 / denominator))
        moments = merged
    count, mean, m2 = moments[0]
    mean = mean.where(count > 0)
    sigma = (m2 / count.where(count > 0)).clip(min=0) ** 0.5
    with ProgressBar():
        return dask.compute(mean, sigma)


def maybe_reorder_levels_by_config(arr: xr.DataArray, order: list | None):
    lvl = "level"
    if (order is None) or (lvl not in arr.dims):
        return arr
    try:
        return arr.sel({lvl: order})
    except Exception:
        return arr


# ----------------- main compute -----------------

def compute_climatology(dataset_files, params, is_pressure_level,
                        chunks="time:64,lat:180,lon:180",
                        consolidated=True,
                        order_levels_by_config=False,
                        progress_mode="vars",
                        force_raw_zarr=False,
                        files_limit=None,
                        agg_mode="pairwise",
                        open_concurrency=1,
                        start="1979-01-01", end="2016-12-31") -> xr.Dataset:
    dask.config.set({
        "array.slicing.split_large_chunks": True,
        "optimization.fuse.active": True,
    })

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    validate_range(start, end)
    dataset_files = [path for path in dataset_files
                     if not (match := re.search(r"(\d{8})\.zarr$", Path(path).name))
                     or start <= pd.Timestamp(match.group(1)) <= end]
    if len(dataset_files) == 0:
        raise FileNotFoundError("No .zarr files found for the dataset.")

    chunk_dict = parse_chunks(chunks)
    mean_list, std_list, name_list = [], [], []

    var_iter = params
    if progress_mode in ("vars", "vars+files"):
        var_iter = tqdm(params, desc="Variables", dynamic_ncols=True)

    for var in var_iter:
        if agg_mode == "concat":
            mean_v, std_v = concat_mode(
                dataset_files, var, chunk_dict, consolidated,
                show_file_tqdm=(progress_mode == "vars+files"),
                force_raw_zarr=force_raw_zarr,
                files_limit=files_limit,
                open_concurrency=open_concurrency, start=start, end=end,
            )
        else:  # pairwise
            mean_v, std_v = pairwise_mode(
                dataset_files, var, chunk_dict, consolidated,
                show_file_tqdm=(progress_mode == "vars+files"),
                force_raw_zarr=force_raw_zarr,
                files_limit=files_limit,
                open_concurrency=open_concurrency, start=start, end=end,
            )

        if mean_v is None or not bool(np.isfinite(mean_v).all()) or not bool(np.isfinite(std_v).all()):
            raise ValueError(f"Variable {var}: empty or invalid statistics")

        if is_pressure_level and ("level" in mean_v.dims) and (mean_v.ndim == 1):
            if order_levels_by_config:
                order = getattr(config, "PRESSURE_LEVELS", None)
                mean_v = maybe_reorder_levels_by_config(mean_v, order)
                std_v = maybe_reorder_levels_by_config(std_v, order)

            for lev, mv, sv in zip(mean_v["level"].values.tolist(),
                                   mean_v.values.tolist(),
                                   std_v.values.tolist()):
                name_list.append(f"{var}-{lev}")
                mean_list.append(float(mv))
                std_list.append(float(sv))
            continue

        name_list.append(var)
        mean_list.append(float(mean_v.values))
        std_list.append(float(std_v.values))

    if not name_list:
        raise RuntimeError("No variables produced results. Check params and inputs.")

    ds_out = xr.Dataset(
        data_vars={"mean": ("param", mean_list), "sigma": ("param", std_list)},
        coords={"param": ("param", name_list)},
        attrs={"note": f"mean/sigma over (time, lat, lon); agg={agg_mode}; raw_zarr={force_raw_zarr}",
               "period_start": str(start.date()), "period_end": str(end.date()), "ddof": 0,
               "spatial_weighting": "unweighted grid points", "accumulation_dtype": "float64",
               "missing_values": "skip NaN; downloader replaces NaN with zero"},
    )
    return ds_out


def main(args):
    # ---- Dask scheduler ----
    if args.scheduler == "distributed":
        try:
            from dask.distributed import Client, LocalCluster
            nworkers = args.n_workers or max(1, (os.cpu_count() or 8) // 2)
            cluster = LocalCluster(
                n_workers=nworkers,
                threads_per_worker=args.threads_per_worker,
                processes=True,
                memory_limit=args.memory_limit,
                dashboard_address=":0",
            )
            client = Client(cluster)
            logging.info(f"Dask dashboard: {client.dashboard_link}")
        except Exception as e:
            logging.warning(f"Failed to start distributed scheduler: {e}; fallback to threaded scheduler.")
            dask.config.set(scheduler="threads")
    else:
        dask.config.set(scheduler="threads")

    # ---- dataset & params ----
    if args.dataset_name == "pressure_level_1.5":
        data_dir = Path(config.DATA_DIR) / "pressure_level_1.5"
        params = list(getattr(config, "ERA5_PRESSURE_LEVEL"))
        is_pl = True
    elif args.dataset_name == "single_level_1.5":
        data_dir = Path(config.DATA_DIR) / "single_level_1.5"
        params = list(getattr(config, "ERA5_SINGLE_LEVEL"))
        is_pl = False
    else:
        raise ValueError("dataset_name must be 'single_level_1.5' or 'pressure_level_1.5'.")

    dataset_files = sorted(data_dir.glob("*.zarr"))
    if args.files_limit and args.files_limit > 0:
        logging.info(f"Found {len(dataset_files)} .zarr shards (limiting to {args.files_limit}).")
    else:
        logging.info(f"Found {len(dataset_files)} .zarr shards under {data_dir}")

    ds_out = compute_climatology(
        dataset_files=dataset_files,
        params=params,
        is_pressure_level=is_pl,
        chunks=args.chunks,
        consolidated=args.consolidated,
        order_levels_by_config=args.order_levels_by_config,
        progress_mode=args.progress,
        force_raw_zarr=args.force_raw_zarr,
        files_limit=args.files_limit if args.files_limit and args.files_limit > 0 else None,
        agg_mode=args.agg,
        open_concurrency=max(1, args.open_concurrency), start=args.start, end=args.end,
    )

    # ---- save (force overwrite) ----
    out_dir = Path(config.DATA_DIR) / "climatology_1.5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"climatology_{args.dataset_name}{args.out_suffix}.zarr"
    if out_path.exists():
        logging.info(f"Output {out_path} exists -> removing (force overwrite).")
        shutil.rmtree(out_path)
    ds_out.to_zarr(str(out_path), mode="w")
    logging.info(f"✅ Saved climatology to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # dataset
    p.add_argument("--dataset_name", default="single_level_1.5",
                   choices=["single_level_1.5", "pressure_level_1.5"])
    p.add_argument("--out_suffix", default="_new", help='Output filename suffix.')
    # IO & chunks
    p.add_argument("--chunks", default="time:64,lat:180,lon:180",
                   help='Rechunk spec (auto-maps lat/latitude, lon/longitude).')
    p.add_argument("--consolidated", action="store_true",
                   help="Use consolidated zarr metadata if present.")
    p.add_argument("--force_raw_zarr", action="store_true",
                   help="Force raw zarr+dask path for all files (fast when many shards or no .zmetadata).")
    p.add_argument("--open_concurrency", type=int, default=1,
                   help="Thread-pool concurrency for opening Zarr files (I/O parallelism).")
    p.add_argument("--files_limit", type=int, default=0,
                   help="Limit input shards; narrow --start/--end accordingly. 0=all.")
    # aggregation & ordering
    p.add_argument("--agg", default="pairwise", choices=["pairwise", "concat"],
                   help="Aggregation mode: 'pairwise' (faster) or 'concat' (direct mean/std over concatenated time).")
    p.add_argument("--order_levels_by_config", action="store_true",
                   help="Reorder pressure levels by config.PRESSURE_LEVELS (if present).")
    # progress
    p.add_argument("--progress", default="vars", choices=["none", "vars", "vars+files"],
                   help="tqdm progress mode (default: vars).")
    # scheduler
    p.add_argument("--scheduler", default="distributed", choices=["distributed", "threads"],
                   help="Dask scheduler (distributed recommended).")
    p.add_argument("--n_workers", type=int, default=0, help="Number of Dask workers (0=auto).")
    p.add_argument("--threads_per_worker", type=int, default=1, help="Threads per worker.")
    p.add_argument("--memory_limit", default="auto", help='Per-worker memory limit, e.g., "6GB" or "auto".')
    p.add_argument("--start", type=parse_date, default=parse_date("1979-01-01"), help="First training date")
    p.add_argument("--end", type=parse_date, default=parse_date("2016-12-31"), help="Last training date (inclusive)")
    args = p.parse_args()
    validate_range(args.start, args.end)
    main(args)


