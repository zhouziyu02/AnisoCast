#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 CirT 已有的 1.5° ERA5 zarr 作为 Ground Truth，
对 Pangu 2018-01-01 初始化的 D15–D42（weeks 3–4 / weeks 5–6 或逐天）做评估，
输出列结构与 CirT CSV 一致：
['steps','pred_vars','RMSE','MAE','Bias','R2','ACC','MS_SSIM','SpecDiv','SpecRes']
"""

from pathlib import Path
import argparse
import numpy as np
import xarray as xr
import pandas as pd

# ========= 配置 =========
BASE = Path("/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT")

# Pangu 预测（Step3 产生）
PANGU_FORECAST = BASE / "data/forecast/pangu_20180101_d15-42_daily00z.npz"

# CirT 1.5° ERA5 zarr 路径（目录下有 era5_pressure_full_1.5deg_*.zarr / era5_single_full_1.5deg_*.zarr）
PRESSURE_ZARR_DIR = BASE / "data/S2S/pressure_level_1.5"
SINGLE_ZARR_DIR   = BASE / "data/S2S/single_level_1.5"

# 初始化日期（与预测一致）
INIT_DATE = "2018-01-01"

# 输出 CSV
OUT_DIR = BASE / "results/pangu_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "Pangu_full_metrics_20180101.csv"

# 变量/层顺序（与前面流程保持一致）
PL_LEVELS = [1000,925,850,700,600,500,400,300,250,200,150,100,50]  # hPa
PL_VARS   = ["z","q","t","u","v"]  # 映射自 ERA5：geopotential/q/t/u/v
SFC_VARS  = ["msl","u10","v10","t2m"]

# ========= 工具 =========

def xr_open_zarr(path):
    # 有的 zarr 未 consolidate，双分支兼容
    try:
        return xr.open_zarr(path, consolidated=True)
    except Exception:
        return xr.open_zarr(path, consolidated=False)

def _pick_one_zarr(dir_path: Path, prefix: str) -> Path:
    """
    在目录中选择一个 zarr（例如 era5_pressure_full_1.5deg_*.zarr）。
    如果有多份，取排序后的第一个（通常是全集，如 19790101 开头）。
    """
    cands = sorted(dir_path.glob(f"{prefix}_*.zarr"))
    if not cands:
        raise FileNotFoundError(f"未在 {dir_path} 找到 {prefix}_*.zarr")
    return cands[0]

def _find_time_coord(ds: xr.Dataset) -> str:
    """自动找到 datetime64 类型的时间坐标名"""
    for cand in ("time", "valid_time", "datetime", "date"):
        if (cand in ds.dims) or (cand in ds.coords):
            v = ds[cand]
            if np.issubdtype(v.dtype, np.datetime64):
                return cand
    for name in list(ds.dims) + list(ds.coords):
        try:
            v = ds[name]
            if np.issubdtype(v.dtype, np.datetime64):
                return name
        except Exception:
            pass
    raise RuntimeError(f"未找到 datetime64 时间坐标；dims={list(ds.dims)}, coords={list(ds.coords)}")

def _ensure_time_index(ds: xr.Dataset, time_name: str) -> xr.Dataset:
    """
    目标：把 ds 的 time_name 变成 1D 的 DatetimeIndex，并按时间升序排序。
    兼容以下异常情况：
    - time 不是坐标
    - time DataArray 不是 1D（例如有多余维度）
    - 数据集中真正的 1D 时间轴藏在别的变量/坐标里
    """
    import pandas as pd
    import numpy as np
    # 若不是坐标，先设为坐标（有些 zarr 把 time 当普通变量）
    if time_name not in ds.coords:
        ds = ds.set_coords(time_name)

    # 优先：直接使用一个“仅随 time 变化”的 1D datetime64 变量/坐标
    time_1d = None
    # 1) 尝试 ds.coords[time_name] 本身
    try:
        da = ds[time_name]
        if da.ndim == 1 and da.dims == (time_name,) and np.issubdtype(da.dtype, np.datetime64):
            time_1d = da.values
    except Exception:
        pass

    # 2) 在所有 coords 中寻找：dtype 为 datetime64，dims 恰好是 (time_name,)
    if time_1d is None:
        for nm in list(ds.coords):
            da = ds[nm]
            if (time_name in da.dims) and (da.dims == (time_name,)) and np.issubdtype(da.dtype, np.datetime64):
                time_1d = da.values
                break

    # 3) 在 data_vars 中也找一遍（有些数据把时间轴单独存在数据变量里）
    if time_1d is None:
        for nm in list(ds.data_vars):
            da = ds[nm]
            if (time_name in da.dims) and (da.dims == (time_name,)) and np.issubdtype(da.dtype, np.datetime64):
                time_1d = da.values
                break

    # 4) 兜底：把 ds[time_name] 扁平化、去重、排序，构造一条 1D 时间索引
    if time_1d is None:
        da = ds[time_name]
        vals = np.asarray(da.values).reshape(-1)
        # 过滤非 datetime64 的情况，强制转成 datetime64[ns]
        try:
            vals = pd.to_datetime(vals)
        except Exception:
            # 再兜底一次：先转成 numpy datetime64
            vals = vals.astype("datetime64[ns]")
        time_1d = np.unique(vals)  # 去重 + 排序

        # 检查长度是否与维度 size 匹配，不匹配也没关系：xarray 允许 coords 长度等于该维度长度
        # 后续 sel(method="nearest") 以 coords 为准

    # 最终：把它设成坐标并排序
    time_1d = pd.to_datetime(time_1d)
    ds = ds.assign_coords({time_name: time_1d})
    ds = ds.sortby(time_name)
    return ds


def read_truth_1p5(init_date: str):
    """
    读取 CirT 1.5° 的 ERA5 真值，返回：
      arr_pl: (28, 5, 13, H, W)  对应 [z,q,t,u,v] × 13层
      arr_sf: (28, 4, H, W)      对应 [msl,u10,v10,t2m]
    取样时间：init_date 的 D15–D42，每天 00Z（若源是 6h 数据，nearest 容忍 3h）
    """
    # 自动在目录下选择 zarr
    pz = _pick_one_zarr(PRESSURE_ZARR_DIR, "era5_pressure_full_1.5deg")
    sz = _pick_one_zarr(SINGLE_ZARR_DIR,   "era5_single_full_1.5deg")
    import pandas as pd
    ti_pl = pd.DatetimeIndex(ds_pl[time_name_pl].values)
    ti_sf = pd.DatetimeIndex(ds_sf[time_name_sf].values)

    idx_pl = ti_pl.get_indexer(days, method="nearest")
    idx_sf = ti_sf.get_indexer(days, method="nearest")

    # 安全检查：若出现 -1 说明索引器失败（一般不会发生，因为我们已排序）
    if (idx_pl < 0).any():
        raise RuntimeError(f"pressure 时间轴最近邻失败：{idx_pl}")
    if (idx_sf < 0).any():
        raise RuntimeError(f"single 时间轴最近邻失败：{idx_sf}")

    ds_pl = ds_pl.isel({time_name_pl: idx_pl})
    ds_sf = ds_sf.isel({time_name_sf: idx_sf})

    # 自动识别时间轴并修复为有序索引
    time_name_pl = _find_time_coord(ds_pl)
    time_name_sf = _find_time_coord(ds_sf)
    ds_pl = _ensure_time_index(ds_pl, time_name_pl)
    ds_sf = _ensure_time_index(ds_sf, time_name_sf)

    # D15–D42 共 28 天
    t0 = np.datetime64(init_date + "T00:00:00")
    days = [t0 + np.timedelta64(d, "D") for d in range(15, 43)]

    # 最近邻对齐 00Z（容忍 3 小时），兼容 6h/3h 数据
    ds_pl = ds_pl.sel({time_name_pl: days}, method="nearest", tolerance=np.timedelta64(3, "h"))
    ds_sf = ds_sf.sel({time_name_sf: days}, method="nearest", tolerance=np.timedelta64(3, "h"))

    # pressure 变量映射
    rename_map = {
        "geopotential":"z",
        "specific_humidity":"q",
        "temperature":"t",
        "u_component_of_wind":"u",
        "v_component_of_wind":"v",
    }
    if not all(k in ds_pl.data_vars for k in rename_map):
        raise RuntimeError(f"pressure zarr 缺少变量，现有: {list(ds_pl.data_vars)}")
    dss_pl = ds_pl[list(rename_map)].rename(rename_map)
    level_name = "level" if "level" in dss_pl.dims else ("isobaricInhPa" if "isobaricInhPa" in dss_pl.dims else None)
    if level_name is None:
        raise RuntimeError(f"pressure 未找到气压层维度，dims={list(dss_pl.dims)}")

    lat = "latitude" if "latitude" in dss_pl.dims else ("lat" if "lat" in dss_pl.dims else None)
    lon = "longitude" if "longitude" in dss_pl.dims else ("lon" if "lon" in dss_pl.dims else None)
    if lat is None or lon is None:
        raise RuntimeError(f"pressure 未找到经纬度维度，dims={list(dss_pl.dims)}")

    dss_pl = dss_pl.sortby(level_name)
    if np.any(np.diff(dss_pl[lat].values) > 0):  # 若由南到北，改为北到南
        dss_pl = dss_pl.sortby(lat, ascending=False)

    levels = dss_pl[level_name].values
    idx = [int(np.where(levels == L)[0][0]) for L in PL_LEVELS]
    arr_pl = dss_pl.to_array().transpose(time_name_pl, "variable", level_name, lat, lon).values
    arr_pl = arr_pl[:, :, idx, :, :].astype(np.float32)  # (28,5,13,H,W)

    # single
    def pick(ds, a, b): return a if a in ds.data_vars else b
    name_msl = pick(ds_sf, "msl", "mean_sea_level_pressure")
    name_u10 = pick(ds_sf, "u10", "10m_u_component_of_wind")
    name_v10 = pick(ds_sf, "v10", "10m_v_component_of_wind")
    name_t2m = pick(ds_sf, "t2m", "2m_temperature")
    dss_sf = xr.Dataset({"msl": ds_sf[name_msl], "u10": ds_sf[name_u10],
                         "v10": ds_sf[name_v10], "t2m": ds_sf[name_t2m]})
    lat2 = "latitude" if "latitude" in dss_sf.dims else ("lat" if "lat" in dss_sf.dims else None)
    lon2 = "longitude" if "longitude" in dss_sf.dims else ("lon" if "lon" in dss_sf.dims else None)
    if lat2 is None or lon2 is None:
        raise RuntimeError(f"single 未找到经纬度维度，dims={list(dss_sf.dims)}")
    if np.any(np.diff(dss_sf[lat2].values) > 0):
        dss_sf = dss_sf.sortby(lat2, ascending=False)

    arr_sf = dss_sf[["msl","u10","v10","t2m"]].to_array().transpose(time_name_sf, "variable", lat2, lon2).values
    arr_sf = arr_sf.astype(np.float32)

    return arr_pl, arr_sf  # (28,5,13,H,W), (28,4,H,W)

def block_mean(arr, bh, bw):
    """对最后两个维度做块平均（整形+平均）；如不整除，自动裁边。"""
    H, W = arr.shape[-2], arr.shape[-1]
    H2, W2 = (H // bh) * bh, (W // bw) * bw
    if H2 != H or W2 != W:
        arr = arr[..., :H2, :W2]
    return arr.reshape(*arr.shape[:-2], H2//bh, bh, W2//bw, bw).mean(axis=(-3,-1))

def downsample_to_1p5(pl025, sf025):
    """0.25° → 1.5°（6×6 块平均）"""
    return block_mean(pl025, 6, 6), block_mean(sf025, 6, 6)

def area_weight(lat_size):
    """纬度余弦权重（形状 (H,1)）"""
    lats = np.linspace(90, -90, lat_size)
    w = np.cos(np.deg2rad(lats))
    w[w < 0] = 0
    return w[:, None]

def _weighted_rmse(a, b, w):
    d = a - b
    return float(np.sqrt(np.sum(w * d * d) / np.sum(w)))

def _weighted_mae(a, b, w):
    d = np.abs(a - b)
    return float(np.sum(w * d) / np.sum(w))

def _weighted_bias(a, b, w):
    d = a - b
    return float(np.sum(w * d) / np.sum(w))

def r2_window(pred, truth):
    a = pred.reshape(-1); b = truth.reshape(-1)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((b - b.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def acc_window(pred, truth):
    a = pred.reshape(-1) - pred.mean()
    b = truth.reshape(-1) - truth.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0

# —— MS-SSIM：优先用 skimage；如不可用则返回 NaN（仍保持列格式一致） ——
def ms_ssim_window(pred, truth):
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return float("nan")
    T = pred.shape[0]
    vals = []
    for t in range(T):
        y = truth[t]; x = pred[t]
        rng = float(max(y.max() - y.min(), 1e-6))
        vals.append(float(ssim(y, x, data_range=rng, gaussian_weights=True, use_sample_covariance=False)))
    return float(np.mean(vals))

# —— 频谱指标：2D rFFT 径向功率谱的对称 KL-like 距离 + 高频能量比 ——
from numpy.fft import rfft2, rfftfreq
def spectrum_metrics(pred, truth):
    def radial_spectrum(field):
        F = rfft2(field)
        P = (F * np.conj(F)).real
        H, W = field.shape
        ky = np.fft.fftfreq(H)
        kx = rfftfreq(W)
        KY, KX = np.meshgrid(ky, kx, indexing="ij")
        kr = np.sqrt(KX**2 + KY**2)
        bins = np.linspace(0, kr.max(), 50)
        spec = np.zeros(len(bins)-1, dtype=np.float64)
        for i in range(len(bins)-1):
            mask = (kr >= bins[i]) & (kr < bins[i+1])
            if mask.any():
                spec[i] = P[mask].mean()
        spec += 1e-12
        spec /= spec.sum()
        return spec

    T = pred.shape[0]
    divs, highs = [], []
    for t in range(T):
        sp_t = radial_spectrum(truth[t])
        sp_p = radial_spectrum(pred[t])
        kl1 = np.sum(sp_p * np.log(sp_p / sp_t))
        kl2 = np.sum(sp_t * np.log(sp_t / sp_p))
        divs.append(0.5 * (kl1 + kl2))
        n = len(sp_t); cut = int(0.8 * n)
        highs.append(float(sp_p[cut:].sum() / max(sp_t[cut:].sum(), 1e-12)))
    return float(np.mean(divs)), float(np.mean(highs))

def evaluate_window(yhat, y, w_lat):
    """
    对窗口 (T,H,W) 计算：
    RMSE/MAE/Bias（先按时间步做加权，再对时间平均） + R2/ACC/MS_SSIM/SpecDiv/SpecRes（窗口整体）
    """
    T = y.shape[0]
    rmse_t, mae_t, bias_t = [], [], []
    for t in range(T):
        rmse_t.append(_weighted_rmse(yhat[t], y[t], w_lat))
        mae_t.append(_weighted_mae(yhat[t], y[t], w_lat))
        bias_t.append(_weighted_bias(yhat[t], y[t], w_lat))
    R2  = r2_window(yhat, y)
    ACC = acc_window(yhat, y)
    MS  = ms_ssim_window(yhat, y)
    SD, SR = spectrum_metrics(yhat, y)
    return (np.mean(rmse_t), np.mean(mae_t), np.mean(bias_t), R2, ACC, MS, SD, SR)

# ========= 主流程 =========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["window","daily"], default="window",
                        help="window=两周窗口汇总(steps=0→w3-4, 1→w5-6); daily=逐天(steps=0..27)")
    parser.add_argument("--out", default=str(OUT_CSV), help="输出CSV路径")
    args = parser.parse_args()

    # 读取 CirT·1.5° ERA5 真值
    truth_pl, truth_sf = read_truth_1p5(INIT_DATE)  # (28,5,13,H,W), (28,4,H,W)

    # 读取 Pangu 预测（0.25°），下采样到 1.5°
    npz = np.load(PANGU_FORECAST)
    pred_pl_025 = npz["pl"]   # (28,5,13,721,1440)
    pred_sf_025 = npz["sfc"]  # (28,4,721,1440)
    pred_pl, pred_sf = downsample_to_1p5(pred_pl_025, pred_sf_025)  # -> (28,*,*,120,240)

    # 空间尺寸对齐（如有 120/121 或 240/241 差异，统一裁到较小者）
    Ht, Wt = truth_pl.shape[-2], truth_pl.shape[-1]
    Hp, Wp = pred_pl.shape[-2],  pred_pl.shape[-1]
    H2, W2 = min(Ht, Hp), min(Wt, Wp)
    truth_pl  = truth_pl[..., :H2, :W2]
    truth_sf  = truth_sf[..., :H2, :W2]
    pred_pl   = pred_pl[..., :H2, :W2]
    pred_sf   = pred_sf[..., :H2, :W2]

    w_lat = area_weight(H2)
    rows = []

    def add_row(step_idx, name, vals):
        RMSE, MAE, Bias, R2, ACC, MS, SD, SR = vals
        rows.append({
            "steps": int(step_idx),
            "pred_vars": name,
            "RMSE": float(RMSE),
            "MAE": float(MAE),
            "Bias": float(Bias),
            "R2": float(R2),
            "ACC": float(ACC),
            "MS_SSIM": float(MS),
            "SpecDiv": float(SD),
            "SpecRes": float(SR),
        })

    if args.mode == "window":
        # 窗口：weeks3–4 (D15–28) / weeks5–6 (D29–42)
        W34 = slice(0, 14)
        W56 = slice(14, 28)
        # 气压层
        for vi, vname in enumerate(PL_VARS):
            for li, level in enumerate(PL_LEVELS):
                name = f"{vname}-{level}"
                vals34 = evaluate_window(pred_pl[W34, vi, li], truth_pl[W34, vi, li], w_lat)
                add_row(0, name, vals34)
                vals56 = evaluate_window(pred_pl[W56, vi, li], truth_pl[W56, vi, li], w_lat)
                add_row(1, name, vals56)
        # 单层
        for vi, vname in enumerate(SFC_VARS):
            vals34 = evaluate_window(pred_sf[W34, vi], truth_sf[W34, vi], w_lat)
            add_row(0, vname, vals34)
            vals56 = evaluate_window(pred_sf[W56, vi], truth_sf[W56, vi], w_lat)
            add_row(1, vname, vals56)

    else:  # daily 模式：逐天（steps=0..27 对应 D15..D42）
        T = pred_pl.shape[0]  # 28
        for t in range(T):
            # 气压层
            for vi, vname in enumerate(PL_VARS):
                for li, level in enumerate(PL_LEVELS):
                    name = f"{vname}-{level}"
                    vals = evaluate_window(pred_pl[t:t+1, vi, li], truth_pl[t:t+1, vi, li], w_lat)
                    add_row(t, name, vals)
            # 单层
            for vi, vname in enumerate(SFC_VARS):
                vals = evaluate_window(pred_sf[t:t+1, vi], truth_sf[t:t+1, vi], w_lat)
                add_row(t, vname, vals)

    df = pd.DataFrame(rows, columns=[
        "steps","pred_vars","RMSE","MAE","Bias","R2","ACC","MS_SSIM","SpecDiv","SpecRes"
    ])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ 已保存：{out_path}")

    # 简要汇总
    if args.mode == "window":
        for k, win in [(0, "weeks3-4"), (1, "weeks5-6")]:
            sub = df[df["steps"] == k]
            print(f"{win} 平均： RMSE={sub['RMSE'].mean():.4f}  MAE={sub['MAE'].mean():.4f}  R2={sub['R2'].mean():.4f}  ACC={sub['ACC'].mean():.4f}  MS-SSIM={sub['MS_SSIM'].mean():.4f}")
    else:
        print("daily 模式：你可以按 steps=0..13 归为 w3-4、14..27 归为 w5-6 自行聚合统计。")

if __name__ == "__main__":
    main()
