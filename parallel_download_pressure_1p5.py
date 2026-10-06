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

# 可选：抑制 xarray 后端插件告警（cfgrib 未安装的提示）
import warnings
warnings.filterwarnings("ignore", message="Engine 'cfgrib' loading failed")

# 全局只读 Dataset（在 main 里初始化）
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
    """
    用全局只读 Dataset 处理并保存单天：
    - 选择当天 + 指定变量 + 指定等压面
    - 0.25° -> 1.5°（步长6）降采样
    - 保存为本地 zarr（默认强制覆盖）
    """
    from dask import config as dask_config  # 延迟导入以减小主进程开销

    # 避免线程内再并发，防止资源过度竞争
    dask_config.set(scheduler="synchronous")

    global GLOBAL_DS
    ds = GLOBAL_DS
    if ds is None:
        raise RuntimeError("GLOBAL_DS is not initialized")

    date_str = date.strftime("%Y-%m-%d")
    ymd = date.strftime("%Y%m%d")
    out_path = os.path.join(out_dir, f"era5_pressure_full_1.5deg_{ymd}.zarr")

    if os.path.exists(out_path) and overwrite:
        shutil.rmtree(out_path, ignore_errors=True)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            sub = ds.sel(time=date_str)[era5_pressure_vars]
            sub = sub.sel(level=pressure_levels)

            # 0.25° -> 1.5°
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
        description="Low-memory parallel download & downsample ERA5 pressure-level daily data to 1.5°."
    )
    p.add_argument("--url", type=str, default=DEFAULT_URL,
                   help="Zarr store URL on GCS (gs://...)")
    p.add_argument("--start", type=str, default="2017-01-01",
                   help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default="2018-12-31",
                   help="End date (YYYY-MM-DD) (inclusive)")
    p.add_argument("--workers", type=int, default=128,
                   help="Max concurrent workers (建议 2–6，根据内存调整)")
    p.add_argument("--retries", type=int, default=3,
                   help="Retries per day")
    p.add_argument("--downsample_factor", type=int, default=6,
                   help="0.25° -> 1.5° 的步长（默认6）")
    p.add_argument("--output_dir", type=str, default=None,
                   help='Output dir (默认: f"{config.DATA_DIR}/pressure_level_1.5")')
    p.add_argument("--anon", action="store_true",
                   help="Use anonymous access for public GCS (recommended for weatherbench2)")
    p.add_argument("--no-overwrite", action="store_true",
                   help="若设置，则遇到已存在的输出会报错而不是覆盖（默认会覆盖）。")
    return p.parse_args()


def main():
    import fsspec

    args = parse_args()

    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    dates = pd.date_range(start_date, end_date, freq="D")

    output_dir = args.output_dir or os.path.join(config.DATA_DIR, "pressure_level_1.5")
    os.makedirs(output_dir, exist_ok=True)

    era5_pressure_vars = config.ERA5_PRESSURE_LEVEL
    pressure_levels    = config.PRESSURE_LEVELS

    storage_options = {"token": "anon"} if args.anon else None

    # —— 关键改动：只在这里打开一次 zarr —— #
    # consolidated=True 读取合并元数据，减少小文件访问
    # chunks="auto" 保持底层块；也可考虑 {'time': 1} 进一步降低单次内存峰值
    global GLOBAL_DS
    GLOBAL_DS = xr.open_zarr(
        args.url,
        consolidated=True,
        chunks=None,                # ← 关闭 dask，避开 SciPy 依赖
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

    # 关闭全局 Dataset（释放文件句柄）
    try:
        GLOBAL_DS.close()
    except Exception:
        pass

    print(f"\nDone. Success: {ok}, Failed: {fail}, Output dir: {output_dir}")


if __name__ == "__main__":
    # 强烈建议限制底层数值库线程，避免过度并发
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    main()
