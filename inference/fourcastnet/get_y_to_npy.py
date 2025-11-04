#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# >>> 关键修复：按你的工程结构导入 S2SDataset
# /mnt/.../CirT/CIRT/dataset.py
from CIRT.dataset_new import S2SDataset
from CIRT.config import *

def main(args):
    # 读取配置
    with open(args.config_filepath, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.FullLoader)
    model_args = hyperparams["model_args"]
    data_args = hyperparams["data_args"]

    print(data_args["test_years"][0])

    # 输出目录：./results/<model_name> 与 ./<year>_y/<model_name>
    model_name = model_args.get("model_name", args.model_name)
    results_dir = Path(f"./results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    year0 = str(data_args["test_years"][0])
    out_dir = Path(f"/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构造测试数据集（注意：此处是 Dataset 本体，不是 DataLoader）
    test_dataset = S2SDataset(
        data_dir=data_args["data_dir"],
        years=data_args["test_years"],
        n_step=data_args["n_step"],
        lead_time=data_args["lead_time"],
        era5_vars=data_args["era5_vars"],
        lra5_vars=data_args["lra5_vars"],
        oras5_vars=data_args["oras5_vars"],
        pred_era5_vars=data_args["pred_era5_vars"],
        pred_lra5_vars=data_args["pred_lra5_vars"],
        pred_oras5_vars=data_args["pred_oras5_vars"],
    )

    # 逐样本保存 y 为 .npy；时间按 UTC 命名到“日-小时”
    for i in range(len(test_dataset)):
        timestamp, x, y = test_dataset[i]

        # timestamp 若为纳秒（你原代码写的是 /1e9），保留该逻辑：
        ts_sec = float(timestamp) / 1e9
        dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
        date_str = dt_utc.strftime("%Y%m%d%H")  # 含小时，避免重名

        # y 可能在 GPU；转 CPU 再保存
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)

        np.save(out_dir / f"{date_str}.npy", y_np)
        # 可选：打印进度
        # print(date_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", default="/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/CIRT/configs/fourcastnetv2.yaml")
    parser.add_argument("--model_name", default="fourcastnetv2")
    args = parser.parse_args()
    main(args)
