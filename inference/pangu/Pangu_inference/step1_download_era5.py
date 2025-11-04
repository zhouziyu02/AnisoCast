#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cdsapi
from pathlib import Path

# ====== 基本设置（可按需修改）======
DATE = "2018-01-01"
TIME = "00:00"

# 下载到你指定的 data 目录
OUTDIR = Path("/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/data/era5_init")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Pangu 需要的气压层与变量
pressure_levels = ["1000","925","850","700","600","500","400","300","250","200","150","100","50"]
variables_pl = ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"]

# 地面变量（单层）
variables_sfc = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature"]

def main():
    # 密钥从 ~/.cdsapirc 读取（不要把 key 写进代码）
    # 如遇企业证书问题，可改成：c = cdsapi.Client(verify=0)
    c = cdsapi.Client()

    print(f"==> 将 ERA5 初始场下载到：{OUTDIR}")

    # 气压层初始场
    print("==> 下载 ERA5 气压层初始场 ...")
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": variables_pl,
            "pressure_level": pressure_levels,
            "year": DATE.split("-")[0],
            "month": DATE.split("-")[1],
            "day": DATE.split("-")[2],
            "time": TIME,
            "area": [90, 0, -90, 359.75],  # 全球
        },
        str(OUTDIR / "initial_pl.grib"),
    )

    # 单层初始场
    print("==> 下载 ERA5 单层初始场 ...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": variables_sfc,
            "year": DATE.split("-")[0],
            "month": DATE.split("-")[1],
            "day": DATE.split("-")[2],
            "time": TIME,
            "area": [90, 0, -90, 359.75],
        },
        str(OUTDIR / "initial_sfc.grib"),
    )

    print("✅ 下载完成：", OUTDIR)

if __name__ == "__main__":
    main()
