#!/usr/bin/env bash

set -euo pipefail

self_name=$(basename "$0")
root_dir=$(cd "$(dirname "$0")" && pwd)
log_dir="$root_dir/logs"
mkdir -p "$log_dir"

# ---------------------------------------------------------------
# 默认参数 (可通过 CLI 覆盖)
# ---------------------------------------------------------------
learning_rate="1e-3"
weight_decay="1e-5"
epochs=20
t_max=500
grad_clip_norm=1.0
embed_dim=768
depth=8
decoder_depth=2
num_heads=16
mlp_ratio=4.0
drop_path=0.1
drop_rate=0.1
batch_size=32
num_workers=16
pred_len=2
np=${NP:-8}
use_tensorboard=false
background=true
custom_tag=""

# 固定数据配置
img_size_h=121
img_size_w=240
input_size=63
output_size=63
data_dir='/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/data/S2S'
train_years=(1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016)
val_years=(2017)
test_years=(2018)
n_step=28
lead_time=15
single_vars=('10m_u_component_of_wind' '10m_v_component_of_wind' '2m_temperature')
pred_single_vars=('10m_u_component_of_wind' '10m_v_component_of_wind' '2m_temperature')
pred_pressure_vars=('geopotential' 'specific_humidity' 'temperature' 'u_component_of_wind' 'v_component_of_wind' 'vertical_velocity')

usage() {
  cat <<EOF
Usage: bash $self_name [options]

Options:
  --lr, --learning-rate <float>     学习率 (示例: --lr 5e-4 或 --lr5e-4)
  --batch, --batch-size <int>       批次大小
  --epochs <int>                    训练轮数
  --embed, --embed-dim <int>        嵌入维度 (需能整除注意力头数)
  --depth <int>                     Transformer层数
  --decoder-depth <int>             解码器层数
  --heads, --num-heads <int>        注意力头数
  --mlp-ratio <float>               MLP扩展比例
  --drop-path <float>               DropPath率
  --drop-rate <float>               Dropout率
  --weight-decay <float>            权重衰减
  --grad-clip <float>               梯度裁剪阈值
  --np <int>                        GPU数量
  --tensorboard                     启用TensorBoard
  --no-tensorboard                  禁用TensorBoard
  --foreground                      前台运行 (默认后台)
  --tag <string>                    自定义标签 (写入日志和PID)
  -h, --help                        查看帮助
EOF
}

while [[ $# -gt 0 ]]; do
  arg="$1"
  case $arg in
    -h|--help)
      usage
      exit 0
      ;;
    --lr|--learning-rate)
      learning_rate="$2"; shift 2 ;;
    --lr*)
      learning_rate="${arg#--lr}"; shift ;;
    --batch|--batch-size|--batchsize)
      batch_size="$2"; shift 2 ;;
    --batch*)
      batch_size="${arg#--batch}"; shift ;;
    --epochs)
      epochs="$2"; shift 2 ;;
    --embed|--embed-dim)
      embed_dim="$2"; shift 2 ;;
    --depth)
      depth="$2"; shift 2 ;;
    --decoder-depth)
      decoder_depth="$2"; shift 2 ;;
    --heads|--num-heads)
      num_heads="$2"; shift 2 ;;
    --mlp-ratio)
      mlp_ratio="$2"; shift 2 ;;
    --drop-path)
      drop_path="$2"; shift 2 ;;
    --drop-rate)
      drop_rate="$2"; shift 2 ;;
    --weight-decay)
      weight_decay="$2"; shift 2 ;;
    --grad-clip)
      grad_clip_norm="$2"; shift 2 ;;
    --np)
      np="$2"; shift 2 ;;
    --tensorboard)
      use_tensorboard=true; shift ;;
    --no-tensorboard)
      use_tensorboard=false; shift ;;
    --foreground)
      background=false; shift ;;
    --background)
      background=true; shift ;;
    --tag)
      custom_tag="$2"; shift 2 ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

array_to_yaml_list() {
  local arr=("$@")
  local result="["
  for i in "${!arr[@]}"; do
    [[ $i -gt 0 ]] && result+=", "
    result+="${arr[$i]}"
  done
  result+="]"
  printf '%s' "$result"
}

str_array_to_yaml_list() {
  local arr=("$@")
  local result="["
  for i in "${!arr[@]}"; do
    [[ $i -gt 0 ]] && result+=", "
    result+="'${arr[$i]}'"
  done
  result+="]"
  printf '%s' "$result"
}

# 为当前运行生成时间戳，并在logs目录中创建稳定的配置文件路径
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$log_dir/ours_${timestamp}.log"
config_file="$log_dir/ours_${timestamp}.yaml"

cat > "$config_file" <<EOF
model_args:
    model_name: 'ours'
    input_size: $input_size
    output_size: $output_size
    learning_rate: $learning_rate
    weight_decay: $weight_decay
    num_workers: $num_workers
    epochs: $epochs
    t_max: $t_max
    pred_len: $pred_len
    grad_clip_norm: $grad_clip_norm
    only_headline: False
    img_size: [$img_size_h, $img_size_w]
    embed_dim: $embed_dim
    depth: $depth
    decoder_depth: $decoder_depth
    num_heads: $num_heads
    mlp_ratio: $mlp_ratio
    drop_path: $drop_path
    drop_rate: $drop_rate

data_args:
    batch_size: $batch_size
    train_years: $(array_to_yaml_list "${train_years[@]}")
    test_years: $(array_to_yaml_list "${test_years[@]}")
    val_years: $(array_to_yaml_list "${val_years[@]}")
    data_dir: '$data_dir'
    n_step: $n_step
    lead_time: $lead_time
    single_vars: $(str_array_to_yaml_list "${single_vars[@]}")
    pred_single_vars: $(str_array_to_yaml_list "${pred_single_vars[@]}")
    pred_pressure_vars: $(str_array_to_yaml_list "${pred_pressure_vars[@]}")
EOF

settings_snapshot=$(cat <<SET
lr=$learning_rate
batch_size=$batch_size
epochs=$epochs
embed_dim=$embed_dim
depth=$depth
decoder_depth=$decoder_depth
num_heads=$num_heads
mlp_ratio=$mlp_ratio
drop_path=$drop_path
drop_rate=$drop_rate
weight_decay=$weight_decay
grad_clip_norm=$grad_clip_norm
np=$np
tag=$custom_tag
SET
)

echo "=========================================="
echo "🚀 Ours 模型训练"
echo "=========================================="
echo "$settings_snapshot"
echo "=========================================="
echo
export NP=$np

common_args=(--config_filepath "$config_file" --devices "$np" --accelerator gpu)
if (( np > 1 )); then
  common_args+=(--strategy ddp_find_unused_parameters_true)
fi
$use_tensorboard && common_args+=(--use_tensorboard)

if (( np > 1 )); then
  launch_cmd=(torchrun --standalone --nproc_per_node="$np" train.py)
else
  launch_cmd=(python3 -u train.py)
fi
launch_cmd+=("${common_args[@]}")

find_checkpoint_by_config() {
  local config_path="$1"
  local start_time="$2"
  
  # 计算当前config文件的特征值（用于匹配）
  local config_hash=$(md5sum "$config_path" 2>/dev/null | cut -d' ' -f1 || md5 -q "$config_path" 2>/dev/null)
  
  # 查找lightning_logs中所有version目录（按时间倒序，最新的在前）
  local version_dirs=($(find lightning_logs -maxdepth 1 -type d -name "version_*" -printf '%T@ %p\n' 2>/dev/null | \
                        sort -rn | cut -d' ' -f2- | head -20))
  
  # 如果find不支持-printf，使用ls
  if [[ ${#version_dirs[@]} -eq 0 ]]; then
    version_dirs=($(ls -td lightning_logs/version_* 2>/dev/null | head -20))
  fi
  
  # 方法1: 通过config.yaml内容匹配（最可靠）
  for version_dir in "${version_dirs[@]}"; do
    local version_config="$version_dir/config.yaml"
    if [[ -f "$version_config" ]]; then
      local version_hash=$(md5sum "$version_config" 2>/dev/null | cut -d' ' -f1 || md5 -q "$version_config" 2>/dev/null)
      
      # 如果config文件内容相同
      if [[ "$config_hash" == "$version_hash" ]]; then
        # 查找该目录下的checkpoint
        local checkpoints=($(find "$version_dir/checkpoints" -name "*.ckpt" -type f 2>/dev/null | sort -r))
        if [[ ${#checkpoints[@]} -gt 0 ]]; then
          echo "${checkpoints[0]}"
          return 0
        fi
      fi
    fi
  done
  
  # 方法2: 通过时间戳匹配（如果config内容匹配失败）
  for version_dir in "${version_dirs[@]}"; do
    local version_config="$version_dir/config.yaml"
    if [[ -f "$version_config" ]]; then
      local version_mtime=$(stat -c %Y "$version_config" 2>/dev/null || stat -f %m "$version_config" 2>/dev/null)
      
      # 如果version的config.yaml创建时间在训练开始之后（允许5分钟误差）
      if [[ -n "$version_mtime" && $version_mtime -ge $((start_time - 300)) ]]; then
        local checkpoints=($(find "$version_dir/checkpoints" -name "*.ckpt" -type f 2>/dev/null | sort -r))
        if [[ ${#checkpoints[@]} -gt 0 ]]; then
          # 检查checkpoint的修改时间是否在训练开始之后
          local ckpt_mtime=$(stat -c %Y "${checkpoints[0]}" 2>/dev/null || stat -f %m "${checkpoints[0]}" 2>/dev/null)
          if [[ -n "$ckpt_mtime" && $ckpt_mtime -ge $start_time ]]; then
            echo "${checkpoints[0]}"
            return 0
          fi
        fi
      fi
    fi
  done
  
  return 1
}

run_training_and_eval() {
  local training_start_time=$(date +%s)
  
  {
    echo "== Launch Command =="
    printf ' %q' "${launch_cmd[@]}"
    echo
    echo
    "${launch_cmd[@]}"
    train_exit_code=$?
    echo
    echo "Training exit code: $train_exit_code"
    
    if [[ $train_exit_code -eq 0 ]]; then
      echo "🔍 查找当前训练任务对应的checkpoint..."
      
      # 查找对应的checkpoint
      local found_checkpoint=$(find_checkpoint_by_config "$config_file" "$training_start_time")
      
      if [[ -n "$found_checkpoint" && -f "$found_checkpoint" ]]; then
        echo "✅ 找到checkpoint: $found_checkpoint"
        echo "🎯 开始自动评估 ours 模型..."
        python3 auto_evaluate.py --model_type ours --config_file "$config_file" --checkpoint_path "$found_checkpoint"
        return $?
      else
        echo "⚠️  未找到对应的checkpoint，使用自动查找模式..."
        python3 auto_evaluate.py --model_type ours --config_file "$config_file"
        return $?
      fi
    else
      echo "❌ 训练失败"
      return $train_exit_code
    fi
  } | tee "$log_file"
  return ${PIPESTATUS[0]}
}

if $background; then
  run_training_and_eval > "$log_file" 2>&1 &
  bg_pid=$!
  pid_file="$log_dir/ours_${timestamp}.pid"
  {
    echo "pid=$bg_pid"
    echo "log=$log_file"
    echo "config=$config_file"
    echo "start_time=$(date +%s)"
    [[ -n $custom_tag ]] && echo "tag=$custom_tag"
    echo "settings<<EOF"
    echo "$settings_snapshot"
    echo "EOF"
  } > "$pid_file"
  echo "✅ Training started in background"
  echo "📝 Log: $log_file"
  echo "🆔 PID: $bg_pid"
  echo "📄 Settings recorded in: $pid_file"
else
  run_training_and_eval
fi


