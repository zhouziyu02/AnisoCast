#!/usr/bin/env bash
# ============================================================================
# Ours模型训练和评估脚本
# 用途：统一管理所有可调参数，便于超参搜索
# 使用方法：修改下面的参数，然后运行 bash train_ours.sh
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 1. 训练超参数 (Training Hyperparameters)
# ============================================================================
LEARNING_RATE=0.001          # 学习率 (建议范围: 0.0001 - 0.01)
WEIGHT_DECAY=1e-5            # 权重衰减 (建议范围: 1e-6 - 1e-3)
EPOCHS=20                     # 训练轮数
T_MAX=500                     # 余弦退火调度器的T_max
GRAD_CLIP_NORM=1.0           # 梯度裁剪阈值 (0表示不裁剪)

# ============================================================================
# 2. 模型架构参数 (Model Architecture Parameters)
# ============================================================================
# 这些参数对应 ours.py 中 Model.__init__ 的参数
IMG_SIZE_H=121                # 图像高度 (纬度维度)
IMG_SIZE_W=240               # 图像宽度 (经度维度)
EMBED_DIM=768                # 嵌入维度 (建议: 256, 384, 512, 768)
DEPTH=8                      # Transformer层数 (建议: 4, 6, 8, 12)
DECODER_DEPTH=2              # 解码器层数 (建议: 1, 2, 3)
NUM_HEADS=16                 # 注意力头数 (必须能被embed_dim整除)
MLP_RATIO=4.0                # MLP扩展比例 (建议: 2.0, 4.0, 8.0)
DROP_PATH=0.1                # DropPath率 (建议: 0.0 - 0.3)
DROP_RATE=0.1                # Dropout率 (建议: 0.0 - 0.2)

# ============================================================================
# 3. 数据参数 (Data Parameters)
# ============================================================================
BATCH_SIZE=32                # 批次大小
NUM_WORKERS=16               # 数据加载器工作进程数
INPUT_SIZE=63                # 输入变量数 (pred_pressure_vars*10 + single_vars)
OUTPUT_SIZE=63               # 输出变量数 (pred_pressure_vars*10 + pred_single_vars)
PRED_LEN=2                   # 预测时间步数 (周数)

# 数据路径和年份
DATA_DIR='/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/data/S2S'
TRAIN_YEARS=(1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016)
VAL_YEARS=(2017)
TEST_YEARS=(2018)
N_STEP=28                    # 时间步数
LEAD_TIME=15                 # 提前时间

# 变量列表
SINGLE_VARS=('10m_u_component_of_wind' '10m_v_component_of_wind' '2m_temperature')
PRED_SINGLE_VARS=('10m_u_component_of_wind' '10m_v_component_of_wind' '2m_temperature')
PRED_PRESSURE_VARS=('geopotential' 'specific_humidity' 'temperature' 'u_component_of_wind' 'v_component_of_wind' 'vertical_velocity')

# ============================================================================
# 4. 训练设置 (Training Settings)
# ============================================================================
NP=${NP:-8}                  # GPU数量 (可通过环境变量覆盖)
USE_TENSORBOARD=false        # 是否启用TensorBoard
BACKGROUND=true              # 是否后台运行

# ============================================================================
# 5. 自动生成配置和运行训练
# ============================================================================

# 创建临时配置文件
CONFIG_TEMP=$(mktemp /tmp/ours_config_XXXXXX.yaml)
trap "rm -f $CONFIG_TEMP" EXIT  # 退出时清理临时文件

# 生成YAML配置
# 辅助函数：将数组转换为YAML列表格式
array_to_yaml_list() {
    local arr=("$@")
    local result="["
    for i in "${!arr[@]}"; do
        if [ $i -gt 0 ]; then
            result+=", "
        fi
        result+="${arr[$i]}"
    done
    result+="]"
    echo "$result"
}

# 辅助函数：将字符串数组转换为YAML字符串列表格式
str_array_to_yaml_list() {
    local arr=("$@")
    local result="["
    for i in "${!arr[@]}"; do
        if [ $i -gt 0 ]; then
            result+=", "
        fi
        result+="'${arr[$i]}'"
    done
    result+="]"
    echo "$result"
}

cat > "$CONFIG_TEMP" <<EOF
model_args:
    model_name: 'ours'
    input_size: $INPUT_SIZE
    output_size: $OUTPUT_SIZE
    learning_rate: $LEARNING_RATE
    weight_decay: $WEIGHT_DECAY
    num_workers: $NUM_WORKERS
    epochs: $EPOCHS
    t_max: $T_MAX
    pred_len: $PRED_LEN
    grad_clip_norm: $GRAD_CLIP_NORM
    only_headline: False
    
    # 模型架构参数 (传递给 ours.py 的 Model.__init__)
    img_size: [$IMG_SIZE_H, $IMG_SIZE_W]
    embed_dim: $EMBED_DIM
    depth: $DEPTH
    decoder_depth: $DECODER_DEPTH
    num_heads: $NUM_HEADS
    mlp_ratio: $MLP_RATIO
    drop_path: $DROP_PATH
    drop_rate: $DROP_RATE

data_args:
    batch_size: $BATCH_SIZE
    train_years: $(array_to_yaml_list "${TRAIN_YEARS[@]}")
    test_years: $(array_to_yaml_list "${TEST_YEARS[@]}")
    val_years: $(array_to_yaml_list "${VAL_YEARS[@]}")
    data_dir: '$DATA_DIR'
    n_step: $N_STEP
    lead_time: $LEAD_TIME
    single_vars: $(str_array_to_yaml_list "${SINGLE_VARS[@]}")
    pred_single_vars: $(str_array_to_yaml_list "${PRED_SINGLE_VARS[@]}")
    pred_pressure_vars: $(str_array_to_yaml_list "${PRED_PRESSURE_VARS[@]}")
EOF

echo "=========================================="
echo "🚀 Ours模型训练配置"
echo "=========================================="
echo "📋 训练超参数:"
echo "   - 学习率: $LEARNING_RATE"
echo "   - 权重衰减: $WEIGHT_DECAY"
echo "   - 训练轮数: $EPOCHS"
echo "   - 梯度裁剪: $GRAD_CLIP_NORM"
echo ""
echo "🏗️  模型架构:"
echo "   - 嵌入维度: $EMBED_DIM"
echo "   - 层数: $DEPTH"
echo "   - 解码器层数: $DECODER_DEPTH"
echo "   - 注意力头数: $NUM_HEADS"
echo "   - MLP比例: $MLP_RATIO"
echo "   - DropPath: $DROP_PATH"
echo "   - Dropout: $DROP_RATE"
echo ""
echo "📊 数据设置:"
echo "   - 批次大小: $BATCH_SIZE"
echo "   - GPU数量: $NP"
echo "=========================================="
echo ""

# 设置环境变量
export NP

# 调用原始训练脚本，但使用临时配置文件
CONFIG_FILE="$CONFIG_TEMP"

# 创建日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/ours_${TIMESTAMP}.log"

# 组装训练命令
COMMON_ARGS=(--config_filepath "$CONFIG_FILE" --devices "$NP" --accelerator gpu)
if (( NP > 1 )); then
  COMMON_ARGS+=(--strategy ddp_find_unused_parameters_true)
fi

if [[ "$USE_TENSORBOARD" == "true" ]]; then
  COMMON_ARGS+=(--use_tensorboard)
fi

if (( NP > 1 )); then
  LAUNCH_CMD=(torchrun --standalone --nproc_per_node="${NP}" train.py)
else
  LAUNCH_CMD=(python3 -u train.py)
fi

LAUNCH_CMD+=("${COMMON_ARGS[@]}")

# 运行训练和评估
run_training_and_eval() {
  {
    echo "=========================================="
    echo "Training started at: $(date)"
    echo "Model: ours"
    echo "Config: Generated from train_ours.sh"
    echo "GPUs: $NP"
    echo "TensorBoard: $USE_TENSORBOARD"
    echo "=========================================="
    echo ""
    
    echo "== Effective env =="
    env | grep -E '^(NCCL_|GLOO_|MASTER_|CUDA_VISIBLE_DEVICES=)' || true
    python - <<'PY'
import torch
print("PyTorch:", torch.__version__, " CUDA:", torch.version.cuda)
PY
    echo ""
    
    echo "== Launch Command =="
    printf ' %q' "${LAUNCH_CMD[@]}"
    echo
    echo ""
    
    echo "== Training Output =="
    "${LAUNCH_CMD[@]}" 2>&1
    
    TRAIN_EXIT_CODE=$?
    echo
    echo "=========================================="
    echo "Training finished at: $(date)"
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "=========================================="
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
      echo "✅ 训练完成！"
      
      # 自动评估模型
      echo ""
      echo "🎯 开始自动评估 ours 模型..."
      
      python3 auto_evaluate.py \
          --model_type ours \
          --config_file "$CONFIG_FILE" 2>&1
      
      EVAL_EXIT_CODE=$?
      if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo "✅ ours 评估完成！"
        echo "📁 结果保存在: ./results/ours/"
      else
        echo "❌ ours 评估失败"
      fi
      
      echo
      echo "=========================================="
      echo "All tasks finished at: $(date)"
      echo "=========================================="
      
      return $EVAL_EXIT_CODE
    else
      echo "❌ 训练失败"
      return $TRAIN_EXIT_CODE
    fi
  } | tee "$LOG_FILE"
  
  return ${PIPESTATUS[0]}
}

# 前台或后台运行
if [[ "$BACKGROUND" == "true" ]]; then
  run_training_and_eval > "$LOG_FILE" 2>&1 &
  BG_PID=$!
  PID_FILE="${LOG_DIR}/ours_${TIMESTAMP}.pid"
  echo $BG_PID > "$PID_FILE"
  
  echo "✅ Training started in background"
  echo "📝 Log: $LOG_FILE"
  echo "🆔 PID: $BG_PID"
  echo "📊 Monitor: tail -f $LOG_FILE"
else
  run_training_and_eval
  exit $?
fi

