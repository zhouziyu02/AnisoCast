#!/bin/bash


# 检查Python环境
echo "🔍 检查Python环境..."
python3 --version

# 检查检查点文件是否存在
CHECKPOINT_PATH="/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/lightning_logs/version_86/checkpoints/epoch=13-step=770.ckpt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 检查点文件不存在: $CHECKPOINT_PATH"
    echo "请确认检查点文件路径是否正确"
    exit 1
fi

echo "✅ 检查点文件存在: $CHECKPOINT_PATH"

python3 /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/others/evaluate_vit.py \
    --config_filepath CIRT/configs/ClimaX.yaml \
    --checkpoint_path "$CHECKPOINT_PATH"
    # --save_predictions \
    # --calculate_metrics

echo "✅ 完整评估完成"
echo "📁 结果保存在: ./results/ClimaX/"
echo "📊 包含指标: RMSE, MAE, Bias, R², ACC, MS-SSIM, SpectralDiv, SpectralRes"


