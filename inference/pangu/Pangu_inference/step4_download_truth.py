#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os, time
import cdsapi
import xarray as xr
import numpy as np
import pandas as pd

BASE = Path("/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT")
DATA_DIR = BASE / "data"
TRUTH_DIR = DATA_DIR / "truth_week3_6"
TRUTH_DIR.mkdir(parents=True, exist_ok=True)

# 分片下载保存目录（NetCDF 版）
PL_DIR = TRUTH_DIR / "pl_daily_nc"
SF_DIR = TRUTH_DIR / "sfc_daily_nc"
PL_DIR.mkdir(exist_ok=True)
SF_DIR.mkdir(exist_ok=True)

OUT = TRUTH_DIR / "truth_20180101_d15-42_daily00z.npz"

PL_LEVELS = [1000,925,850,700,600,500,400,300,250,200,150,100,50]
PL_VARS   = ["geopotential","specific_humidity","temperature","u_component_of_wind","v_component_of_wind"]
SFC_TARGET_NAMES = ["msl","u10","v10","t2m"]  # 输出顺序

def get_client():
    # 兼容新旧配置：优先取环境变量，其次 ~/.cdsapirc
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDSAPI_KEY", None)
    verify_env = os.environ.get("CDSAPI_VERIFY")
    verify = True if verify_env is None else bool(int(verify_env))
    if key:
        return cdsapi.Client(url=url, key=key, verify=verify)
    return cdsapi.Client(url=url, verify=verify)

def robust_retrieve(c, dataset, request, target, max_retry=10, sleep_sec=10):
    """下载带重试；不做昂贵的试读校验，只做体积检查。"""
    target = Path(target)
    # 如果有异常小文件，删除重下
    if target.exists() and target.stat().st_size < 100_000:
        try: target.unlink()
        except Exception: pass
    for k in range(1, max_retry+1):
        try:
            c.retrieve(dataset, request, str(target))
            # 体积最小校验（100KB）；避免远端异常返回的 html/json
            if target.exists() and target.stat().st_size >= 100_000:
                return True
            else:
                raise RuntimeError("文件异常小，疑似下载不完整")
        except Exception as e:
            print(f"[{target.name}] 第{k}次下载失败：{e}")
            if target.exists():
                try: target.unlink()
                except Exception: pass
            time.sleep(sleep_sec)
    return False

def xr_open_nc(path):
    """稳健打开 NetCDF：优先 h5netcdf，其次 netcdf4，最后默认。"""
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:
        try:
            return xr.open_dataset(path, engine="netcdf4")
        except Exception:
            return xr.open_dataset(path)  # 兜底：让 xarray 自选

def dl_daily_truth_netcdf():
    c = get_client()
    start = pd.Timestamp("2018-01-01") + pd.Timedelta(days=15)  # 2018-01-16
    days = [start + pd.Timedelta(days=i) for i in range(28)]
    print("==> 按日下载 ERA5（NetCDF）压力层 & 单层 真值（每日 00Z，共 28 天）")
    for d in days:
        y, m, dd = d.strftime("%Y"), d.strftime("%m"), d.strftime("%d")
        fpl = PL_DIR / f"pl_{y}{m}{dd}_00.nc"
        fsf = SF_DIR / f"sfc_{y}{m}{dd}_00.nc"

        if not fpl.exists():
            req_pl = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": PL_VARS,
                "pressure_level": [str(l) for l in PL_LEVELS],
                "year": y, "month": m, "day": dd, "time": "00:00",
            }
            ok = robust_retrieve(c, "reanalysis-era5-pressure-levels", req_pl, fpl)
            if not ok: raise RuntimeError(f"下载失败（压力层）: {fpl}")
        else:
            print(f"[跳过] 已存在 {fpl.name}")

        if not fsf.exists():
            req_sf = {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": ["mean_sea_level_pressure","10m_u_component_of_wind",
                             "10m_v_component_of_wind","2m_temperature"],
                "year": y, "month": m, "day": dd, "time": "00:00",
            }
            ok = robust_retrieve(c, "reanalysis-era5-single-levels", req_sf, fsf)
            if not ok: raise RuntimeError(f"下载失败（单层）: {fsf}")
        else:
            print(f"[跳过] 已存在 {fsf.name}")

def read_pl_stack_nc():
    """ 读取 NetCDF 压力层 (T, 5, 13, 721, 1440) """
    arrs = []
    for p in sorted(PL_DIR.glob("pl_*_00.nc")):
        ds = xr_open_nc(p)
        # 变量名到短名统一
        rename_map = {
            "geopotential":"z",
            "specific_humidity":"q",
            "temperature":"t",
            "u_component_of_wind":"u",
            "v_component_of_wind":"v",
        }
        need = [k for k in rename_map if k in ds.data_vars]
        if len(need) != 5:
            raise RuntimeError(f"{p.name} 变量不全：{list(ds.data_vars)}")
        dss = ds[need].rename(rename_map)

        level_name = "level" if "level" in dss.dims else ("isobaricInhPa" if "isobaricInhPa" in dss.dims else None)
        if level_name is None:
            raise RuntimeError(f"{p.name} 未找到气压层维度，dims={list(dss.dims)}")

        lat = "latitude" if "latitude" in dss.dims else ("lat" if "lat" in dss.dims else None)
        lon = "longitude" if "longitude" in dss.dims else ("lon" if "lon" in dss.dims else None)
        if not lat or not lon:
            raise RuntimeError(f"{p.name} 未找到经纬度维度，dims={list(dss.dims)}")

        # 层顺序 & 纬度方向
        dss = dss.sortby(level_name)
        if np.any(np.diff(dss[lat].values) > 0):  # 由南到北？改为北到南
            dss = dss.sortby(lat, ascending=False)

        levels = dss[level_name].values
        idx = [int(np.where(levels == L)[0][0]) for L in PL_LEVELS]
        a = dss.to_array().transpose("variable", level_name, lat, lon).values
        a = a[:, idx, :, :].astype(np.float32)  # (5,13,721,1440)
        arrs.append(a)
        ds.close()
    return np.stack(arrs, axis=0)

def read_sf_stack_nc():
    """ 读取 NetCDF 单层 (T, 4, 721, 1440)，输出 msl/u10/v10/t2m """
    arrs = []
    for p in sorted(SF_DIR.glob("sfc_*_00.nc")):
        ds = xr_open_nc(p)
        name_msl = "msl" if "msl" in ds.data_vars else ("mean_sea_level_pressure" if "mean_sea_level_pressure" in ds.data_vars else None)
        name_u10 = "u10" if "u10" in ds.data_vars else ("10m_u_component_of_wind" if "10m_u_component_of_wind" in ds.data_vars else None)
        name_v10 = "v10" if "v10" in ds.data_vars else ("10m_v_component_of_wind" if "10m_v_component_of_wind" in ds.data_vars else None)
        name_t2m = "t2m" if "t2m" in ds.data_vars else ("2m_temperature" if "2m_temperature" in ds.data_vars else None)
        if None in (name_msl, name_u10, name_v10, name_t2m):
            raise RuntimeError(f"{p.name} 缺少单层变量，vars={list(ds.data_vars)}")

        dss = xr.Dataset({"msl": ds[name_msl], "u10": ds[name_u10],
                          "v10": ds[name_v10], "t2m": ds[name_t2m]})
        lat = "latitude" if "latitude" in dss.dims else ("lat" if "lat" in dss.dims else None)
        lon = "longitude" if "longitude" in dss.dims else ("lon" if "lon" in dss.dims else None)
        if np.any(np.diff(dss[lat].values) > 0):
            dss = dss.sortby(lat, ascending=False)

        a = dss[SFC_TARGET_NAMES].to_array().transpose("variable", lat, lon).values
        arrs.append(a.astype(np.float32))
        ds.close()
    return np.stack(arrs, axis=0)

def main():
    # 若旧的超大 GRIB 合并文件存在，仅提示，不再使用
    for big in [TRUTH_DIR/"truth_pl_d15-42.grib", TRUTH_DIR/"truth_sfc_d15-42.grib"]:
        if big.exists():
            print("⚠️ 检测到历史 GRIB 大文件：", big, "（不再使用，可自行删除节省空间）")

    dl_daily_truth_netcdf()            # 按日 NetCDF 下载
    print("==> 读取并合并 NetCDF ...")
    arr_pl = read_pl_stack_nc()        # (28,5,13,721,1440)
    arr_sf = read_sf_stack_nc()        # (28,4,721,1440)
    assert arr_pl.shape[0] == 28 and arr_sf.shape[0] == 28
    np.savez_compressed(OUT, pl=arr_pl, sfc=arr_sf)
    print("✅ 真值保存：", OUT)

if __name__ == "__main__":
    main()
