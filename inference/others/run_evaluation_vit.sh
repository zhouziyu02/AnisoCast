#!/bin/bash

echo "🚀 开始ViT模型完整评估..."

# 检查Python环境
echo "🔍 检查Python环境..."
python3 --version

# 检查检查点文件是否存在
CHECKPOINT_PATH="/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/lightning_logs/version_69/checkpoints/epoch=14-step=825.ckpt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 检查点文件不存在: $CHECKPOINT_PATH"
    echo "请确认检查点文件路径是否正确"
    exit 1
fi

echo "✅ 检查点文件存在: $CHECKPOINT_PATH"

# 运行完整评估
echo "🎯 运行ViT模型完整评估..."
python3 /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/others/evaluate_vit.py \
    --config_filepath CIRT/configs/ViT.yaml \
    --checkpoint_path "$CHECKPOINT_PATH"
    # --save_predictions \
    # --calculate_metrics

echo "✅ 完整评估完成"
echo "📁 结果保存在: ./results/ViT/"
echo "📊 包含指标: RMSE, MAE, Bias, R², ACC, MS-SSIM, SpectralDiv, SpectralRes"
