#!/bin/bash

echo "🚀 开始Transformer模型完整评估..."

# 检查Python环境
echo "🔍 检查Python环境..."
python3 --version

# 检查检查点文件是否存在
CHECKPOINT_PATH="/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/lightning_logs/version_108/checkpoints/epoch=13-step=1526.ckpt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 检查点文件不存在: $CHECKPOINT_PATH"
    echo "请确认检查点文件路径是否正确"
    exit 1
fi

echo "✅ 检查点文件存在: $CHECKPOINT_PATH"

# 运行完整评估
echo "🎯 运行Transformer模型完整评估..."
python3 /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/others/evaluate_Transformer.py \
    --config_filepath CIRT/configs/Transformer.yaml \
    --checkpoint_path "$CHECKPOINT_PATH"
    # --save_predictions \
    # --calculate_metrics

echo "✅ 完整评估完成"
echo "📁 结果保存在: ./results/Transformer/"
echo "📊 包含指标: RMSE, MAE, Bias, R², ACC, MS-SSIM, SpectralDiv, SpectralRes"
