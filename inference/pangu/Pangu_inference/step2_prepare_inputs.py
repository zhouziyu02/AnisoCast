#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import xarray as xr
import numpy as np

BASE = Path("/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT")
DATA_DIR = BASE / "data"
ERA5_INIT_DIR = DATA_DIR / "era5_init"
OUT_DIR = DATA_DIR / "inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PL_PATH  = ERA5_INIT_DIR / "initial_pl.grib"
SFC_PATH = ERA5_INIT_DIR / "initial_sfc.grib"

# 固定顺序（Pangu 常用）
PL_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PL_VARS   = ["z", "q", "t", "u", "v"]  # geopotential, specific_humidity, temperature, u, v
SFC_VARS  = ["msl", "u10", "v10", "t2m"]  # mean sea level pressure, 10m u, 10m v, 2m t

def open_pl():
    # 读气压层（避免混入别的 typeOfLevel）
    ds = xr.open_dataset(
        PL_PATH,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",  # 每次新建索引，避免冲突
        },
    )
    # 保留需要的变量
    ds = ds[PL_VARS]
    # 确认层顺序
    ds = ds.sortby("isobaricInhPa")
    # 重排成 (var, level, lat, lon)
    arr = ds.to_array().transpose("variable", "isobaricInhPa", "latitude", "longitude").values
    # 统一按指定顺序筛选 level
    level_idx = [np.where(ds.isobaricInhPa.values == L)[0].item() for L in PL_LEVELS]

    arr = arr[:, level_idx, :, :].astype(np.float32)
    return arr  # (5, 13, 721, 1440)


def open_sfc():
    """
    鲁棒读取单层初始场：
    - 分别读取 msl / 10m 风 / 2m 温度（兼容 shortName 差异）
    - 不对 time 排序（单时次没有必要，且部分文件 time 为标量会报错）
    - 统一重命名为 msl / u10 / v10 / t2m
    - 输出 (4, 721, 1440) float32
    """
    import xarray as xr

    def read_ds(filter_keys: dict):
        return xr.open_dataset(
            SFC_PATH,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": filter_keys, "indexpath": ""},
        )

    # 1) msl
    ds_msl = read_ds({"shortName": "msl"})
    name_msl = next((v for v in ds_msl.data_vars if v in ("msl",)), None)
    if name_msl is None:
        raise RuntimeError(f"在 {SFC_PATH} 中未找到 msl（data_vars={list(ds_msl.data_vars)})")
    ds_msl = ds_msl[[name_msl]].rename({name_msl: "msl"})

    # 2) 10m 风（u10 / 10u，v10 / 10v 都可能）
    # 先用 typeOfLevel+level=10 粗筛，再兜底用 shortName
    ds_10 = read_ds({"typeOfLevel": "heightAboveGround", "level": 10})
    cand_u = [v for v in ds_10.data_vars if v in ("u10", "10u")]
    cand_v = [v for v in ds_10.data_vars if v in ("v10", "10v")]
    if not cand_u:
        try:
            ds_10u = read_ds({"shortName": "10u"})
            cand_u = [v for v in ds_10u.data_vars if v in ("10u", "u10")]
            if cand_u:
                ds_10 = xr.merge([ds_10, ds_10u], compat="override", join="outer")
        except Exception:
            pass
    if not cand_v:
        try:
            ds_10v = read_ds({"shortName": "10v"})
            cand_v = [v for v in ds_10v.data_vars if v in ("10v", "v10")]
            if cand_v:
                ds_10 = xr.merge([ds_10, ds_10v], compat="override", join="outer")
        except Exception:
            pass
    if not cand_u:
        raise RuntimeError(f"未找到 10m U 风（候选 u10/10u；data_vars={list(ds_10.data_vars)})")
    if not cand_v:
        raise RuntimeError(f"未找到 10m V 风（候选 v10/10v；data_vars={list(ds_10.data_vars)})")
    name_u, name_v = cand_u[0], cand_v[0]
    ds_u10 = ds_10[[name_u]].rename({name_u: "u10"})
    ds_v10 = ds_10[[name_v]].rename({name_v: "v10"})

    # 3) 2m 温度（t2m / 2t）
    try:
        ds_2 = read_ds({"typeOfLevel": "heightAboveGround", "level": 2})
        cand_t = [v for v in ds_2.data_vars if v in ("t2m", "2t")]
    except Exception:
        ds_2 = None
        cand_t = []
    if not cand_t:
        ds_2t = read_ds({"shortName": "2t"})
        cand_t = [v for v in ds_2t.data_vars if v in ("2t", "t2m")]
        if not cand_t:
            raise RuntimeError(f"未找到 2m 温度（候选 t2m/2t；data_vars={list(ds_2t.data_vars)})")
        ds_2 = ds_2t
    name_t = cand_t[0]
    ds_t2m = ds_2[[name_t]].rename({name_t: "t2m"})

    # 4) 合并（不 sortby time；单时次也可能无 time 维或为标量）
    ds = xr.merge([ds_msl, ds_u10, ds_v10, ds_t2m], compat="override", join="exact")

    # 5) 取第一个时间步；若无 time 维则直接使用
    if "time" in ds.dims:
        ds0 = ds.isel(time=0)
    else:
        ds0 = ds

    # 6) 固定顺序并转 (var, lat, lon)
    arr = ds0[["msl", "u10", "v10", "t2m"]].to_array().transpose("variable", "latitude", "longitude").values
    return arr.astype(np.float32)





def main():
    print("==> 读取气压层 ...")
    fields_pl = open_pl()   # (5,13,lat,lon)
    print("==> 读取单层 ...")
    fields_sfc = open_sfc() # (4,lat,lon)

    # 简单形状检查
    assert fields_pl.shape[0] == 5 and fields_pl.shape[1] == 13, "气压层维度不符"
    assert fields_sfc.shape[0] == 4, "单层变量数不符"

    out_path = OUT_DIR / "pangu_inputs_20180101.npz"
    np.savez_compressed(out_path, pl=fields_pl, sfc=fields_sfc,
                        pl_levels=np.array(PL_LEVELS), pl_vars=np.array(PL_VARS), sfc_vars=np.array(SFC_VARS))
    print("✅ 保存：", out_path)

if __name__ == "__main__":
    main()
