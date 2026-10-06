#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast climatology over (time, lat/lon) for ERA5 WB2 1.5°

Features
- Parallel & distributed with Dask (LocalCluster)
- Parallel Zarr open (thread-pool): --open_concurrency (preserves file order)
- Two aggregation modes:
    * --agg concat   : concat along time -> mean/std over (time,lat,lon)
    * --agg pairwise : (default) per-day spatial reduce to sum/sum2/count,
                       then sum across days -> mean/std (ddof=0), same result as above,
                       but with much smaller task graphs and better throughput
- Raw Zarr fast path (skip xarray metadata): --force_raw_zarr
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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import xarray as xr
import dask
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

import zarr
import dask.array as da

import config  # must define: DATA_DIR, ERA5_SINGLE_LEVEL, ERA5_PRESSURE_LEVEL, PRESSURE_LEVELS (for PL reorder)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
warnings.filterwarnings("ignore")


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

def _infer_dims_from_shape(shape, lengths):
    """
    Prefer ('time','level','latitude','longitude'); if no level then ('time','latitude','longitude').
    """
    def L(n):
        return lengths.get(n)

    candidates = []
    if "level" in lengths:
        candidates += [
            ("time", "level", "latitude", "longitude"),
            ("time", "latitude", "longitude", "level"),
            ("level", "time", "latitude", "longitude"),
        ]
    candidates += [("time", "latitude", "longitude")]

    for cand in candidates:
        dims = tuple({"lat": "latitude", "lon": "longitude"}.get(d, d) for d in cand)
        lens = tuple(L(d) for d in dims)
        if all(l is not None for l in lens) and tuple(lens) == tuple(shape):
            return dims
    return None


def open_var_via_zarr(path: Path, var: str) -> xr.DataArray | None:
    g = zarr.open_group(str(path), mode="r")
    if var not in g:  # shard lacks this variable
        return None
    arr = g[var]  # zarr.Array
    # coordinates
    coords = {}
    lengths = {}
    for cname in ("time", "level", "latitude", "longitude"):
        if cname in g:
            c_arr = g[cname]
            coords[cname] = c_arr[:]  # small arrays -> RAM
            lengths[cname] = c_arr.shape[0]
    dims = _infer_dims_from_shape(arr.shape, lengths)
    if dims is None:
        raise ValueError(f"Cannot infer dims for {var} in {path.name}: shape={arr.shape}, lens={lengths}")
    darr = da.from_zarr(arr)  # lazy
    da_x = xr.DataArray(darr, dims=dims, coords={d: coords[d] for d in dims if d in coords}, name=var)
    # dtype safety
    if da_x.dtype.kind in ("i", "u"):
        da_x = da_x.astype("float32")
    elif str(da_x.dtype) == "float64":
        da_x = da_x.astype("float32")
    return da_x


# ---------- parallel open (preserve file order) ----------

def _open_one(fp: Path, var: str, chunk_dict, consolidated, force_raw_zarr):
    da_x = None
    if not force_raw_zarr:
        try:
            ds = open_zarr_flexible(fp, consolidated=consolidated)
            if var in ds:
                da_x = ds[var]
        except Exception:
            da_x = None
    if da_x is None:
        try:
            da_x = open_var_via_zarr(fp, var)
        except Exception:
            return None

    if da_x is None:
        return None  # shard missing this variable

    # chunks
    this_chunk = adapt_chunks_to_da(chunk_dict, da_x)
    if this_chunk:
        da_x = da_x.chunk(this_chunk)
    # dtype normalize
    if da_x.dtype.kind in ("i", "u"):
        da_x = da_x.astype("float32")
    elif str(da_x.dtype) == "float64":
        da_x = da_x.astype("float32")
    return da_x


def parallel_open_list(files, var, chunk_dict, consolidated, force_raw_zarr,
                       show_file_tqdm=False, open_concurrency=1):
    """
    Open many shards in parallel and return DataArrays in the SAME ORDER as `files`.
    Missing-variable shards are skipped (logged).
    """
    das_ordered = [None] * len(files)
    missing = 0

    if open_concurrency > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=open_concurrency) as ex:
            fut_to_idx = {ex.submit(_open_one, fp, var, chunk_dict, consolidated, force_raw_zarr): i
                          for i, fp in enumerate(files)}
            iterator = as_completed(fut_to_idx)
            if show_file_tqdm:
                iterator = tqdm(iterator, total=len(fut_to_idx), desc=f"{var}: open files",
                                leave=False, dynamic_ncols=True, mininterval=0.5)
            for fut in iterator:
                i = fut_to_idx[fut]
                try:
                    da_x = fut.result()
                except Exception:
                    da_x = None
                if da_x is None:
                    missing += 1
                else:
                    das_ordered[i] = da_x
    else:
        it = enumerate(files)
        if show_file_tqdm:
            it = enumerate(tqdm(files, desc=f"{var}: scan files", leave=False, dynamic_ncols=True))
        for i, fp in it:
            try:
                da_x = _open_one(fp, var, chunk_dict, consolidated, force_raw_zarr)
            except Exception:
                da_x = None
            if da_x is None:
                missing += 1
            else:
                das_ordered[i] = da_x

    # compact while preserving order
    das = [da for da in das_ordered if da is not None]
    if missing:
        logging.info(f"{var}: {missing} shard(s) missing this variable; skipped.")
    return das


# ---------- aggregation modes ----------

def concat_mode(dataset_files, var, chunk_dict, consolidated,
                show_file_tqdm, force_raw_zarr, files_limit, open_concurrency):
    files = dataset_files[:files_limit] if files_limit else dataset_files
    da_list = parallel_open_list(files, var, chunk_dict, consolidated, force_raw_zarr,
                                 show_file_tqdm=show_file_tqdm, open_concurrency=open_concurrency)
    if not da_list:
        return None, None
    da_all = xr.concat(da_list, dim="time")  # lazy concat along time (chronological)
    reduce_dims = dims_to_reduce(da_all)
    if not reduce_dims:
        return None, None
    da_mean = da_all.mean(dim=reduce_dims, skipna=True)
    da_std = da_all.std(dim=reduce_dims, skipna=True)
    with ProgressBar():
        mean_v = da_mean.compute()
        std_v = da_std.compute()
    return mean_v, std_v


def reduce_spatial_sums(da_x: xr.DataArray):
    # sum over spatial dims; keep time(len=1) & level(if any)
    space_dims = [d for d in ("lat", "latitude", "lon", "longitude") if d in da_x.dims]
    s = da_x.sum(dim=space_dims, skipna=True)
    s2 = (da_x * da_x).sum(dim=space_dims, skipna=True)
    cnt = da_x.notnull().sum(dim=space_dims)
    # drop size-1 time
    if "time" in s.dims and s.sizes["time"] == 1:
        s = s.isel(time=0, drop=True)
        s2 = s2.isel(time=0, drop=True)
        cnt = cnt.isel(time=0, drop=True)
    return s, s2, cnt


def pairwise_mode(dataset_files, var, chunk_dict, consolidated,
                  show_file_tqdm, force_raw_zarr, files_limit, open_concurrency):
    files = dataset_files[:files_limit] if files_limit else dataset_files
    da_list = parallel_open_list(files, var, chunk_dict, consolidated, force_raw_zarr,
                                 show_file_tqdm=show_file_tqdm, open_concurrency=open_concurrency)
    if not da_list:
        return None, None

    S_list, S2_list, N_list = [], [], []
    for da_x in da_list:
        s, s2, n = reduce_spatial_sums(da_x)
        S_list.append(s)
        S2_list.append(s2)
        N_list.append(n)

    def stack_sum(arrs):
        if len(arrs) == 1:
            return arrs[0]
        return xr.concat(arrs, dim="files").sum("files")

    S = stack_sum(S_list)
    S2 = stack_sum(S2_list)
    N = stack_sum(N_list)

    mean = S / N
    var = (S2 / N) - (mean * mean)
    # numerical safety
    std = xr.apply_ufunc(lambda a: (a.clip(min=0)) ** 0.5, var)

    with ProgressBar():
        mean_v = mean.compute()
        std_v = std.compute()
    return mean_v, std_v


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
                        open_concurrency=1) -> xr.Dataset:
    dask.config.set({
        "array.slicing.split_large_chunks": True,
        "optimization.fuse.active": True,
    })

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
                open_concurrency=open_concurrency,
            )
        else:  # pairwise
            mean_v, std_v = pairwise_mode(
                dataset_files, var, chunk_dict, consolidated,
                show_file_tqdm=(progress_mode == "vars+files"),
                force_raw_zarr=force_raw_zarr,
                files_limit=files_limit,
                open_concurrency=open_concurrency,
            )

        if mean_v is None:
            logging.error(f"Variable {var}: cannot compute statistics; skip.")
            continue

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
        attrs={"note": f"mean/sigma over (time, lat, lon); agg={agg_mode}; raw_zarr={force_raw_zarr}"},
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
        open_concurrency=max(1, args.open_concurrency),
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
                   help="Limit number of daily .zarr to scan (for quick sanity check). 0=all.")
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
    args = p.parse_args()
    main(args)


