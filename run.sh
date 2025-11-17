# #!/bin/bash

# # Parse arguments
# MODEL_TYPE=${1:-"CirT"}
# USE_TENSORBOARD=${2:-"false"}

# # Validate model type
# if [[ "$MODEL_TYPE" != "CirT" && "$MODEL_TYPE" != "ClimaX" && "$MODEL_TYPE" != "ViT" && "$MODEL_TYPE" != "ClimODE" && "$MODEL_TYPE" != "EGNN" ]]; then
#     echo "Error: Invalid model type '$MODEL_TYPE'"
#     echo "Usage: $0 [CirT|ClimaX|ViT|ClimODE|EGNN] [true|false]"
#     echo "  MODEL_TYPE: CirT, ClimaX, ViT, ClimODE, or EGNN (default: CirT)"
#     echo "  USE_TENSORBOARD: true to enable TensorBoard, false for max performance (default: false)"
#     echo ""
#     echo "Examples:"
#     echo "  $0 CirT                    # Train CirT without TensorBoard (max performance)"
#     echo "  $0 ClimaX true            # Train ClimaX with TensorBoard logging"
#     echo "  $0 ViT false              # Train ViT without TensorBoard (max performance)"
#     echo "  $0 ClimODE true           # Train ClimODE with TensorBoard logging"
#     echo "  $0 EGNN true              # Train EGNN with TensorBoard logging"
#     exit 1
# fi

# # Set config file based on model type
# CONFIG_FILE="CIRT/configs/${MODEL_TYPE}.yaml"

# echo "🚀 Starting training with model: $MODEL_TYPE"
# echo "📁 Using config file: $CONFIG_FILE"
# echo "📊 TensorBoard logging: $USE_TENSORBOARD"

# # Build command
# CMD="python3 -u train.py --config_filepath \"$CONFIG_FILE\""

# if [[ "$USE_TENSORBOARD" == "true" ]]; then
#     CMD="$CMD --use_tensorboard"
#     echo "📈 TensorBoard logs will be saved to: logs/$MODEL_TYPE"
# else
#     echo "⚡ Using maximum performance mode (no TensorBoard)"
# fi

# echo ""

# # Run training
# eval $CMD


# # 基本用法
# bash ./run.sh [MODEL_TYPE] [USE_TENSORBOARD]

# # 使用示例
# bash ./run.sh CirT                    # 训练CirT，最高性能（无TensorBoard）
# bash ./run.sh CirT true              # 训练CirT，使用TensorBoard日志
# bash ./run.sh ClimaX false           # 训练ClimaX，最高性能
# bash ./run.sh ViT true               # 训练ViT，使用TensorBoard日志
# bash ./run.sh ClimODE true           # 训练ClimODE，使用TensorBoard日志
# bash ./run.sh EGNN false              # 训练EGNN，使用TensorBoard日志

# --

#!/usr/bin/env bash
# run_lo_ddp.sh
# set -euo pipefail

# ########## 0) 网络与 NCCL 基本设置（单机、多卡、仅 loopback 可见的环境） ##########
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_SOCKET_IFNAME=lo
# export GLOO_SOCKET_IFNAME=lo
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# # 本地 rendezvous
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=${MASTER_PORT:-29400}   # 如端口占用，可改成 29401/29402 等

# ########## 1) H20 张量核/TF32 建议 ##########
# export NVIDIA_TF32_OVERRIDE=0   # 允许 TF32
# python - <<'PY'
# import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.set_float32_matmul_precision('high')  # 或 'medium'
# print("TF32:", torch.backends.cuda.matmul.allow_tf32,
#       "fp32 matmul precision:", torch.get_float32_matmul_precision())
# PY

########## 2) 读取参数 ##########
MODEL_TYPE=${1:-"EGNN"}
USE_TENSORBOARD=${2:-"false"}
NP=${NP:-8}                          # 本机 GPU 数；如 4 卡就 NP=4 bash run_lo_ddp.sh
MAIN=${MAIN:-"train.py"}             # 入口脚本
EXTRA=${EXTRA:-""}                   # 额外透传参数（可选）

# 校验模型名
case "$MODEL_TYPE" in
  CirT|ClimaX|ViT|ClimODE|EGNN|FNO|TelePiT|Transformer) ;;
  *)
    echo "Error: Invalid model type '$MODEL_TYPE'"
    echo "Usage: $0 [CirT|ClimaX|ViT|ClimODE|EGNN|FNO|TelePiT|Transformer] [true|false]"
    exit 1
    ;;
esac



CONFIG_FILE="CIRT/configs/${MODEL_TYPE}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi

echo "🚀 Starting training with model: $MODEL_TYPE"
echo "📁 Using config file: $CONFIG_FILE"
echo "📊 TensorBoard logging: $USE_TENSORBOARD"
echo "🧩 GPUs (nproc_per_node): $NP"
echo

########## 3) 组装命令 ##########
BASE_CMD="python3 -u ${MAIN} --config_filepath \"$CONFIG_FILE\""
if [[ "$USE_TENSORBOARD" == "true" ]]; then
  BASE_CMD="$BASE_CMD --use_tensorboard"
  echo "📈 TensorBoard 将启用（可能略降性能）"
else
  echo "⚡ 最大化性能：关闭 TensorBoard"
fi
if [[ -n "$EXTRA" ]]; then
  BASE_CMD="$BASE_CMD $EXTRA"
fi

# Lightning 能自己拉起子进程，但在很多环境下用 torchrun 更稳
CMD="torchrun --standalone --nproc_per_node=${NP} $MAIN --config_filepath \"$CONFIG_FILE\""
if [[ "$USE_TENSORBOARD" == "true" ]]; then
  CMD="$CMD --use_tensorboard"
fi
if [[ -n "$EXTRA" ]]; then
  CMD="$CMD $EXTRA"
fi

########## 4) 打印关键信息并启动 ##########
echo "== Effective env =="
env | grep -E '^(NCCL_|GLOO_|MASTER_|CUDA_VISIBLE_DEVICES=)' || true
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, " CUDA:", torch.version.cuda)
PY
echo

echo "== Launch =="
# 你也可以把下一行换回：eval "$BASE_CMD"
eval "$BASE_CMD"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "✅ 训练完成！"
    
    # 自动评估模型
    echo ""
    echo "🎯 开始自动评估 $MODEL_TYPE 模型..."
    
    # 使用自动评估脚本
    python3 auto_evaluate.py \
        --model_type "$MODEL_TYPE" \
        --config_file "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ $MODEL_TYPE 评估完成！"
        echo "📁 结果保存在: ./results/$MODEL_TYPE/"
        echo "📊 包含指标: RMSE, MAE, Bias, R², ACC, MS-SSIM, SpectralDiv, SpectralRes"
    else
        echo "❌ $MODEL_TYPE 评估失败"
    fi
else
    echo "❌ 训练失败"
    exit 1
fi
