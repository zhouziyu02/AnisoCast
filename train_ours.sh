#!/usr/bin/env bash
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET

set -euo pipefail

self_name=$(basename "$0")
root_dir=$(cd "$(dirname "$0")" && pwd)
log_dir=""  # 将在参数解析后设置

# ---------------------------------------------------------------
# 默认参数 (可通过 CLI 覆盖)
# ---------------------------------------------------------------
learning_rate="1e-3"
weight_decay="1e-5"
epochs=20
t_max=500
grad_clip_norm=1.0
embed_dim=256
patch_size=124
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
model_name="soon"  # 默认使用SOON模型

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
  --patch, --patch-size <int>       patch大小
  --depth <int>                     Transformer层数
  --decoder-depth <int>             解码器层数
  --heads, --num-heads <int>        注意力头数
  --mlp-ratio <float>               MLP扩展比例
  --drop-path <float>               DropPath率
  --drop-rate <float>               Dropout率
  --pred-len <int>                  预测步长
  --t-max <int>                     Cosine调度器T_max
  --weight-decay <float>            权重衰减
  --grad-clip <float>               梯度裁剪阈值
  --np <int>                        GPU数量
  --tensorboard                     启用TensorBoard
  --no-tensorboard                  禁用TensorBoard
  --foreground                      前台运行 (默认后台)
  --tag <string>                    自定义标签 (写入日志和PID)
  --model-name <string>             模型名称 (soon，默认: soon)
  --log-dir <string>                自定义日志目录 (默认: ./logs)
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
    --patch|--patch-size)
      patch_size="$2"; shift 2 ;;
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
    --pred-len)
      pred_len="$2"; shift 2 ;;
    --t-max)
      t_max="$2"; shift 2 ;;
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
    --model-name)
      model_name="$2"; shift 2 ;;
    --log-dir)
      log_dir="$2"; shift 2 ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

# 设置默认日志目录
if [[ -z "$log_dir" ]]; then
  # 如果模型是 soon，默认使用 ./logs/soon
  if [[ "$model_name" == "soon" ]]; then
    log_dir="$root_dir/logs/soon"
  else
    log_dir="$root_dir/logs"
  fi
fi
mkdir -p "$log_dir"

# ---------------------------------------------------------------
# 基本合法性检查
# ---------------------------------------------------------------

# 验证模型名称（支持 soon）
if [[ "$model_name" != "soon" ]]; then
  echo "❌ 错误: 不支持的模型名称 '$model_name'。支持: soon" >&2
  exit 1
fi

# 可选：检查 CUDA_VISIBLE_DEVICES 与 --np 一致性（避免误配）
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _cudas <<< "$CUDA_VISIBLE_DEVICES"
  visible_n=${#_cudas[@]}
  if (( np > visible_n )); then
    echo "❌ 错误: --np=$np 但 CUDA_VISIBLE_DEVICES 只暴露了 $visible_n 张卡: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------
# DDP 端口防冲突（单机多任务/后台并行必备）
# ---------------------------------------------------------------
if (( np > 1 )); then
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

  # 如果外部没指定 MASTER_PORT，则自动生成一个低冲突高位端口
  if [[ -z "${MASTER_PORT:-}" ]]; then
    # 使用 epoch 秒 + 当前脚本 PID 生成一个较稳定且低冲突的端口
    # 范围: 20000 - 39999
    base=$(( (10#$(date +%s) + $$) % 20000 ))
    export MASTER_PORT=$((20000 + base))
  fi
fi

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

# 为当前运行生成文件名和目录（优先使用tag，否则使用时间戳）
timestamp=$(date +"%Y%m%d_%H%M%S")

# 如果提供了tag，使用tag作为子目录名和文件名；否则使用model_name作为目录，时间戳作为文件名
if [[ -n "$custom_tag" ]]; then
  model_log_dir="$log_dir/$custom_tag"
  file_basename="$custom_tag"
else
  model_log_dir="$log_dir/$model_name"
  file_basename="${model_name}_${timestamp}"
fi

mkdir -p "$model_log_dir"
log_file="$model_log_dir/${file_basename}.log"
config_file="$model_log_dir/${file_basename}.yaml"
pid_file="$model_log_dir/${file_basename}.pid"

runtime_strategy="auto"
if (( np > 1 )); then
  runtime_strategy="ddp_find_unused_parameters_true"
fi

cat > "$config_file" <<EOF
model_args:
    model_name: '$model_name'
    input_size: $input_size
    output_size: $output_size
    img_size: [$img_size_h, $img_size_w]
    patch_size: $patch_size
    learning_rate: $learning_rate
    weight_decay: $weight_decay
    num_workers: $num_workers
    epochs: $epochs
    t_max: $t_max
    pred_len: $pred_len
    grad_clip_norm: $grad_clip_norm
    only_headline: False
    embed_dim: $embed_dim
    depth: $depth
    decoder_depth: $decoder_depth
    num_heads: $num_heads
    mlp_ratio: $mlp_ratio
    drop_path: $drop_path
    drop_rate: $drop_rate

data_args:
    batch_size: $batch_size
    num_workers: $num_workers
    train_years: $(array_to_yaml_list "${train_years[@]}")
    test_years: $(array_to_yaml_list "${test_years[@]}")
    val_years: $(array_to_yaml_list "${val_years[@]}")
    data_dir: '$data_dir'
    n_step: $n_step
    lead_time: $lead_time
    single_vars: $(str_array_to_yaml_list "${single_vars[@]}")
    pred_single_vars: $(str_array_to_yaml_list "${pred_single_vars[@]}")
    pred_pressure_vars: $(str_array_to_yaml_list "${pred_pressure_vars[@]}")

train_args:
    batch_size: $batch_size
    learning_rate: $learning_rate
    weight_decay: $weight_decay
    grad_clip_norm: $grad_clip_norm
    epochs: $epochs
    t_max: $t_max
    pred_len: $pred_len
    drop_path: $drop_path
    drop_rate: $drop_rate
    use_tensorboard: $use_tensorboard

runtime_args:
    devices: $np
    accelerator: gpu
    strategy: $runtime_strategy
    background: $background
    tag: '$custom_tag'
    launcher: '$self_name'
EOF

settings_snapshot=$(cat <<SET
lr=$learning_rate
batch_size=$batch_size
epochs=$epochs
patch_size=$patch_size
embed_dim=$embed_dim
depth=$depth
decoder_depth=$decoder_depth
num_heads=$num_heads
mlp_ratio=$mlp_ratio
drop_path=$drop_path
drop_rate=$drop_rate
t_max=$t_max
pred_len=$pred_len
weight_decay=$weight_decay
grad_clip_norm=$grad_clip_norm
np=$np
tag=$custom_tag
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-}
SET
)

echo "=========================================="
echo "🚀 ${model_name} 模型训练"
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
  # 显式指定 master_port，避免同机多任务端口冲突
  launch_cmd=(torchrun --standalone --nproc_per_node="$np" --master_port="$MASTER_PORT" train.py)
else
  launch_cmd=(python3 -u train.py)
fi
launch_cmd+=("${common_args[@]}")

find_checkpoint_by_config() {
  local config_path="$1"
  local start_time="$2"
  local expected_model_name="$3"  # 新增：期望的模型名称

  # 计算当前config文件的特征值（用于匹配）
  local config_hash
  config_hash=$(md5sum "$config_path" 2>/dev/null | cut -d' ' -f1 || md5 -q "$config_path" 2>/dev/null)

  # 从config文件中提取model_name（用于双重验证）
  local config_model_name
  config_model_name=$(grep -E "^model_name:" "$config_path" 2>/dev/null | head -1 | sed -E "s/.*model_name:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/" | tr -d ' ')

  # 查找lightning_logs中所有version目录（按时间倒序，最新的在前）
  local version_dirs=()
  mapfile -t version_dirs < <(find lightning_logs -maxdepth 1 -type d -name "version_*" -printf '%T@ %p\n' 2>/dev/null | \
                        sort -rn | cut -d' ' -f2- | head -20)

  # 如果find不支持-printf，使用ls
  if [[ ${#version_dirs[@]} -eq 0 ]]; then
    mapfile -t version_dirs < <(ls -td lightning_logs/version_* 2>/dev/null | head -20)
  fi

  # 方法1: 通过config.yaml内容匹配 + model_name验证 + 时间戳验证（最可靠）
  for version_dir in "${version_dirs[@]}"; do
    local version_config="$version_dir/config.yaml"
    if [[ -f "$version_config" ]]; then
      local version_hash
      version_hash=$(md5sum "$version_config" 2>/dev/null | cut -d' ' -f1 || md5 -q "$version_config" 2>/dev/null)

      # 如果config文件内容相同
      if [[ "$config_hash" == "$version_hash" ]]; then
        # 验证version目录中的model_name是否匹配（双重验证）
        local version_model_name
        version_model_name=$(grep -E "^model_name:" "$version_config" 2>/dev/null | head -1 | sed -E "s/.*model_name:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/" | tr -d ' ')

        # 如果提供了期望的model_name，必须匹配
        if [[ -n "$expected_model_name" && -n "$version_model_name" ]]; then
          if [[ "$expected_model_name" != "$version_model_name" ]]; then
            continue  # model_name不匹配，跳过这个version目录
          fi
        fi

        # 验证时间戳：config.yaml的创建时间应该在训练开始之后（允许5分钟误差）
        local version_mtime
        version_mtime=$(stat -c %Y "$version_config" 2>/dev/null || stat -f %m "$version_config" 2>/dev/null)
        if [[ -n "$version_mtime" && $version_mtime -lt $((start_time - 300)) ]]; then
          continue  # 时间戳不匹配，跳过
        fi

        # 查找该目录下的checkpoint
        # 优先查找最佳checkpoint（排除last.ckpt，因为现在只保存最佳模型）
        local best_checkpoint
        best_checkpoint=$(find "$version_dir/checkpoints" -name "epoch=*-step=*.ckpt" -type f 2>/dev/null | sort -r | head -1)
        if [[ -n "$best_checkpoint" && -f "$best_checkpoint" ]]; then
          # 验证checkpoint的修改时间应该在训练开始之后
          local ckpt_mtime
          ckpt_mtime=$(stat -c %Y "$best_checkpoint" 2>/dev/null || stat -f %m "$best_checkpoint" 2>/dev/null)
          if [[ -n "$ckpt_mtime" && $ckpt_mtime -ge $start_time ]]; then
            echo "$best_checkpoint"
            return 0
          fi
        fi

        # 如果没有找到epoch=*-step=*.ckpt格式的，回退到查找所有.ckpt文件
        local checkpoints=()
        mapfile -t checkpoints < <(find "$version_dir/checkpoints" -name "*.ckpt" -type f 2>/dev/null | sort -r)
        if [[ ${#checkpoints[@]} -gt 0 ]]; then
          local ckpt_mtime
          ckpt_mtime=$(stat -c %Y "${checkpoints[0]}" 2>/dev/null || stat -f %m "${checkpoints[0]}" 2>/dev/null)
          if [[ -n "$ckpt_mtime" && $ckpt_mtime -ge $start_time ]]; then
            echo "${checkpoints[0]}"
            return 0
          fi
        fi
      fi
    fi
  done

  # 方法2: 通过时间戳 + model_name匹配（如果config内容匹配失败）
  for version_dir in "${version_dirs[@]}"; do
    local version_config="$version_dir/config.yaml"
    if [[ -f "$version_config" ]]; then
      local version_mtime
      version_mtime=$(stat -c %Y "$version_config" 2>/dev/null || stat -f %m "$version_config" 2>/dev/null)

      # 如果version的config.yaml创建时间在训练开始之后（允许5分钟误差）
      if [[ -n "$version_mtime" && $version_mtime -ge $((start_time - 300)) ]]; then
        # 验证model_name是否匹配
        if [[ -n "$expected_model_name" ]]; then
          local version_model_name
          version_model_name=$(grep -E "^model_name:" "$version_config" 2>/dev/null | head -1 | sed -E "s/.*model_name:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/" | tr -d ' ')
          if [[ -n "$version_model_name" && "$expected_model_name" != "$version_model_name" ]]; then
            continue  # model_name不匹配，跳过
          fi
        fi

        # 优先查找最佳checkpoint（排除last.ckpt）
        local best_checkpoint
        best_checkpoint=$(find "$version_dir/checkpoints" -name "epoch=*-step=*.ckpt" -type f 2>/dev/null | sort -r | head -1)
        if [[ -n "$best_checkpoint" && -f "$best_checkpoint" ]]; then
          local ckpt_mtime
          ckpt_mtime=$(stat -c %Y "$best_checkpoint" 2>/dev/null || stat -f %m "$best_checkpoint" 2>/dev/null)
          if [[ -n "$ckpt_mtime" && $ckpt_mtime -ge $start_time ]]; then
            echo "$best_checkpoint"
            return 0
          fi
        fi

        # 如果没有找到epoch=*-step=*.ckpt格式的，回退到查找所有.ckpt文件
        local checkpoints=()
        mapfile -t checkpoints < <(find "$version_dir/checkpoints" -name "*.ckpt" -type f 2>/dev/null | sort -r)
        if [[ ${#checkpoints[@]} -gt 0 ]]; then
          local ckpt_mtime
          ckpt_mtime=$(stat -c %Y "${checkpoints[0]}" 2>/dev/null || stat -f %m "${checkpoints[0]}" 2>/dev/null)
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
  local training_start_time
  training_start_time=$(date +%s)
  local checkpoint_version_dir=""  # 用于存储checkpoint的version目录路径
  local checkpoint_info_file="$config_file.checkpoint_dir"

  # 清理可能存在的旧文件
  rm -f "$checkpoint_info_file"

  {
    echo "== Launch Command =="
    printf ' %q' "${launch_cmd[@]}"
    echo
    echo

    # 运行训练命令，同时捕获checkpoint路径
    "${launch_cmd[@]}" 2>&1 | tee >(while IFS= read -r line; do
      echo "$line"
      # 从训练输出中提取checkpoint目录路径
      if [[ "$line" =~ 💾.*配置已保存到checkpoint目录:\ ([^[:space:]]+) ]]; then
        local extracted_path="${BASH_REMATCH[1]}"
        # 如果是config.yaml路径，转换为version目录
        extracted_path="${extracted_path%/config.yaml}"
        echo "$extracted_path" > "$checkpoint_info_file"
      fi
    done)

    train_exit_code=${PIPESTATUS[0]}
    echo
    echo "Training exit code: $train_exit_code"

    # 读取保存的checkpoint目录路径
    if [[ -f "$checkpoint_info_file" ]]; then
      checkpoint_version_dir=$(cat "$checkpoint_info_file" 2>/dev/null)
      rm -f "$checkpoint_info_file"
    fi

    if [[ $train_exit_code -eq 0 ]]; then
      echo "🔍 查找当前训练任务对应的checkpoint..."
      echo "📋 期望的模型名称: $model_name"

      local found_checkpoint=""

      # 方法1: 如果从训练日志中提取到了checkpoint目录，直接使用
      if [[ -n "$checkpoint_version_dir" && -d "$checkpoint_version_dir" ]]; then
        echo "📁 使用训练时保存的checkpoint目录: $checkpoint_version_dir"
        # 查找该目录下的最佳checkpoint
        found_checkpoint=$(find "$checkpoint_version_dir/checkpoints" -name "epoch=*-step=*.ckpt" -type f 2>/dev/null | sort -r | head -1)
        if [[ -z "$found_checkpoint" || ! -f "$found_checkpoint" ]]; then
          # 如果没有找到epoch=*-step=*.ckpt格式的，查找所有.ckpt文件
          found_checkpoint=$(find "$checkpoint_version_dir/checkpoints" -name "*.ckpt" -type f 2>/dev/null | sort -r | head -1)
        fi
      fi

      # 方法2: 如果方法1失败，使用原来的查找逻辑
      if [[ -z "$found_checkpoint" || ! -f "$found_checkpoint" ]]; then
        echo "⚠️  未从训练日志中提取到checkpoint路径，使用自动查找模式..."
        found_checkpoint=$(find_checkpoint_by_config "$config_file" "$training_start_time" "$model_name")
      fi

      if [[ -n "$found_checkpoint" && -f "$found_checkpoint" ]]; then
        echo "✅ 找到checkpoint: $found_checkpoint"
        echo "🎯 开始自动评估 $model_name 模型..."
        if [[ -n "$custom_tag" ]]; then
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file" --checkpoint_path "$found_checkpoint" --tag "$custom_tag"
        else
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file" --checkpoint_path "$found_checkpoint"
        fi
        return $?
      else
        echo "⚠️  未找到对应的checkpoint，使用自动查找模式..."
        if [[ -n "$custom_tag" ]]; then
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file" --tag "$custom_tag"
        else
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file"
        fi
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
  # 确保日志文件目录存在
  mkdir -p "$(dirname "$log_file")"
  # 创建空日志文件，确保它存在
  touch "$log_file"

  # 后台运行时，所有输出重定向到日志文件，不打印到终端
  run_training_and_eval > "$log_file" 2>&1 &
  bg_pid=$!

  # pid_file已经在上面定义了，使用model_log_dir下的路径
  {
    echo "pid=$bg_pid"
    echo "log=$log_file"
    echo "config=$config_file"
    echo "start_time=$(date +%s)"
    echo "model_name=$model_name"
    [[ -n $custom_tag ]] && echo "tag=$custom_tag"
    echo "MASTER_ADDR=${MASTER_ADDR:-}"
    echo "MASTER_PORT=${MASTER_PORT:-}"
    echo "settings<<EOF"
    echo "$settings_snapshot"
    echo "EOF"
  } > "$pid_file"

  # 只在终端显示启动信息，不显示训练输出
  echo "✅ Training started in background"
  echo "📝 Log: $log_file"
  echo "🆔 PID: $bg_pid"
  echo "📄 Settings recorded in: $pid_file"
  echo "🔍 查看日志: tail -f $log_file"
else
  run_training_and_eval
fi


# 
# bash train_ours.sh --model-name soon --lr 1e-3 --embed 256 --depth 7 --decoder-depth 1 --tag lr1e-3_embed256_depth7_dedep1