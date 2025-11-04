#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import onnxruntime as ort
import sys

BASE = Path("/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT")
DATA_DIR = BASE / "data"
INP = DATA_DIR / "inputs" / "pangu_inputs_20180101.npz"

# ✅ 改这里：使用你已下载的权重目录
ASSETS = BASE / "panguweather_assets"   # e.g. /mnt/.../CirT/panguweather_assets
OUT_DIR = DATA_DIR / "forecast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "pangu_20180101_d15-42_daily00z.npz"

def pick_providers():
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        print("==> 使用 CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("==> 未检测到 CUDA，使用 CPUExecutionProvider（会较慢）")
    return ["CPUExecutionProvider"]

def ensure_fortran_contiguous(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not x.flags["F_CONTIGUOUS"]:
        x = np.asfortranarray(x)
    return x

def locate_weight(dirpath: Path, kind: str) -> Path:
    """
    kind in {"6","24"}; 在 dirpath 下寻找对应 onnx 文件。
    首选标准名：pangu_weather_6.onnx / pangu_weather_24.onnx
    也兼容常见变体名称（包含 6 或 24 的 onnx）。
    """
    preferred = dirpath / f"pangu_weather_{kind}.onnx"
    if preferred.exists():
        return preferred
    # 兼容变体：包含 'pangu' 和 'onnx'，同时文件名里包含 6 或 24
    candidates = sorted([p for p in dirpath.glob("*.onnx")
                         if ("pangu" in p.name.lower() or "panguweather" in p.name.lower())
                         and kind in p.stem])
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"未找到 {kind}h 权重。请确认目录 {dirpath} 下存在 "
        f"'pangu_weather_{kind}.onnx'，或包含 {kind} 的 Pangu onnx 文件。"
    )

def main():
    if not INP.exists():
        print(f"未找到输入：{INP}\n请先完成 Step 2。", file=sys.stderr)
        sys.exit(1)
    if not ASSETS.exists():
        print(f"未找到权重目录：{ASSETS}", file=sys.stderr)
        sys.exit(1)

    print("==> 定位权重文件 ...")
    onnx_6  = locate_weight(ASSETS, "6")
    onnx_24 = locate_weight(ASSETS, "24")
    print(f"   ✓ 6h  权重: {onnx_6}")
    print(f"   ✓ 24h 权重: {onnx_24}")

    print("==> 读取初始张量 ...")
    data = np.load(INP)
    fields_pl  = ensure_fortran_contiguous(data["pl"])   # (5,13,721,1440)
    fields_sfc = ensure_fortran_contiguous(data["sfc"])  # (4,721,1440)

    providers = pick_providers()

    print("==> 加载 ONNX 模型 ...")
    sess6  = ort.InferenceSession(str(onnx_6),  providers=providers)
    sess24 = ort.InferenceSession(str(onnx_24), providers=providers)

    in_main_6  = sess6.get_inputs()[0].name
    in_surf_6  = sess6.get_inputs()[1].name
    in_main_24 = sess24.get_inputs()[0].name
    in_surf_24 = sess24.get_inputs()[1].name

    # 初始状态
    st_pl_6  = fields_pl.copy()
    st_sf_6  = fields_sfc.copy()
    st_pl_24 = fields_pl.copy()
    st_sf_24 = fields_sfc.copy()

    lead_hours = 42 * 24
    step = 6

    save_days = list(range(15, 43))  # 15..42
    saved_pl, saved_sfc = [], []

    print("==> 开始积分 (0–42 天, Δt=6h；24h 用 24h 模型跳步) ...")
    for h in range(step, lead_hours + 1, step):
        if h % 24 == 0:
            out_pl, out_sf = sess24.run(None, {in_main_24: st_pl_24, in_surf_24: st_sf_24})
            st_pl_24 = ensure_fortran_contiguous(out_pl)
            st_sf_24 = ensure_fortran_contiguous(out_sf)
            st_pl_6, st_sf_6 = st_pl_24, st_sf_24
        else:
            out_pl, out_sf = sess6.run(None, {in_main_6: st_pl_6, in_surf_6: st_sf_6})
            st_pl_6 = ensure_fortran_contiguous(out_pl)
            st_sf_6 = ensure_fortran_contiguous(out_sf)

        if h % 24 == 0:
            day = h // 24
            if day in save_days:
                pl = np.asarray(st_pl_6, dtype=np.float32).reshape(5, 13, 721, 1440)
                sf = np.asarray(st_sf_6, dtype=np.float32).reshape(4, 721, 1440)
                saved_pl.append(pl)
                saved_sfc.append(sf)
                print(f"   ✓ 保存 day {day} ({h}h) 输出")

    saved_pl  = np.stack(saved_pl,  axis=0)  # (28,5,13,721,1440)
    saved_sfc = np.stack(saved_sfc, axis=0)  # (28,4,721,1440)
    np.savez_compressed(OUT_PATH, pl=saved_pl, sfc=saved_sfc)
    print("✅ 预测保存：", OUT_PATH)

if __name__ == "__main__":
    main()

python - <<'PY'
import os, pathlib
p = pathlib.Path.home()/'.cdsapirc'
print("将创建配置文件：", p)
apikey = input("请输入你的 CDS API KEY（仅一段，不需要 UID）: ").strip()
need_v = input("公司网络需要关闭证书校验吗？(y/N): ").strip().lower()=='y'
lines = ["url: https://cds.climate.copernicus.eu/api", f"key: {apikey}"]
if need_v: lines.append("verify: 0")
p.write_text("\n".join(lines)+"\n", encoding="utf-8")
os.chmod(p, 0o600)
print("✅ 已写入 ~/.cdsapirc 并设为 600 权限")
PY
