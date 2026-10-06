#!/usr/bin/env bash
set -euo pipefail

self_name=$(basename "$0")
root_dir=$(cd "$(dirname "$0")" && pwd)
cd "$root_dir"
python_bin=${PYTHON:-python3}
log_dir=""  # Will be set after argument parsing

# ---------------------------------------------------------------
# Default parameters (can be overridden via CLI)
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
model_name="soon"  # Default to SOON model

# Fixed data configuration
img_size_h=121
img_size_w=240
input_size=63
output_size=63
data_dir=${ANISOCAST_DATA_DIR:-"$root_dir/data/S2S"}
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
  --lr, --learning-rate <float>     Learning rate (e.g., --lr 5e-4 or --lr5e-4)
  --batch, --batch-size <int>       Batch size
  --epochs <int>                    Number of epochs
  --embed, --embed-dim <int>        Embedding dimension
  --patch, --patch-size <int>       Patch size
  --depth <int>                     Number of operator blocks
  --decoder-depth <int>             Decoder depth
  --heads, --num-heads <int>        Compatibility argument (unused by backbone)
  --mlp-ratio <float>               MLP expansion ratio
  --drop-path <float>               DropPath rate
  --drop-rate <float>               Dropout rate
  --pred-len <int>                  Prediction length
  --t-max <int>                     Cosine scheduler T_max
  --weight-decay <float>            Weight decay
  --grad-clip <float>               Gradient clipping threshold
  --np <int>                        Number of GPUs
  --tensorboard                     Enable TensorBoard
  --no-tensorboard                  Disable TensorBoard
  --foreground                      Run in foreground (default: background)
  --tag <string>                    Custom tag (written to logs and PID)
  --model-name <string>             Model name (soon, default: soon)
  --log-dir <string>                Custom log directory (default: ./logs)
  -h, --help                        Show help
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

# Set default log directory
if [[ -z "$log_dir" ]]; then
  # If model is soon, default to ./logs/soon
  if [[ "$model_name" == "soon" ]]; then
    log_dir="$root_dir/logs/soon"
  else
    log_dir="$root_dir/logs"
  fi
fi
mkdir -p "$log_dir"

# ---------------------------------------------------------------
# Basic validation checks
# ---------------------------------------------------------------

# Validate model name (supports soon)
if [[ "$model_name" != "soon" ]]; then
  echo "Error: Unsupported model name '$model_name'. Supported: soon" >&2
  exit 1
fi

[[ "$np" =~ ^[1-9][0-9]*$ ]] || { echo "--np must be a positive integer" >&2; exit 2; }
[[ "$pred_len" == 2 ]] || { echo "--pred-len must be 2" >&2; exit 2; }
[[ -z "$custom_tag" || ( "$custom_tag" =~ ^[a-zA-Z0-9_.-]+$ && "$custom_tag" != . && "$custom_tag" != .. ) ]] || { echo "Invalid --tag" >&2; exit 2; }

# Optional: Check CUDA_VISIBLE_DEVICES consistency with --np (avoid misconfiguration)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _cudas <<< "$CUDA_VISIBLE_DEVICES"
  visible_n=${#_cudas[@]}
  if (( np > visible_n )); then
    echo "Error: --np=$np but CUDA_VISIBLE_DEVICES only exposes $visible_n GPUs: $CUDA_VISIBLE_DEVICES" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------
# DDP port conflict prevention (required for multi-task/background parallel execution)
# ---------------------------------------------------------------
if (( np > 1 )); then
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

  # If MASTER_PORT is not specified externally, auto-generate a low-conflict high port
  if [[ -z "${MASTER_PORT:-}" ]]; then
    # Use epoch seconds + current script PID to generate a stable and low-conflict port
    # Range: 20000 - 39999
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

# Generate filename and directory for current run (prefer tag, otherwise use timestamp)
timestamp=$(date +"%Y%m%d_%H%M%S")

# If tag is provided, use tag as subdirectory name and filename; otherwise use model_name as directory, timestamp as filename
if [[ -n "$custom_tag" ]]; then
  model_log_dir="$log_dir/$custom_tag"
  file_basename="$custom_tag"
else
  model_log_dir="$log_dir/$model_name"
  file_basename="${model_name}_${timestamp}"
fi

[[ ! -e "$model_log_dir/checkpoints" ]] || { echo "Run already has checkpoints; choose a new --tag" >&2; exit 2; }
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
echo "Training ${model_name} model"
echo "=========================================="
echo "$settings_snapshot"
echo "=========================================="
echo

export NP=$np

common_args=(--config_filepath "$config_file" --run-dir "$model_log_dir" --devices "$np" --accelerator gpu)
if (( np > 1 )); then
  common_args+=(--strategy ddp_find_unused_parameters_true)
fi
$use_tensorboard && common_args+=(--use_tensorboard)

if (( np > 1 )); then
  # Explicitly specify master_port to avoid port conflicts in multi-task scenarios
  launch_cmd=("$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$np" --master_port="$MASTER_PORT" train.py)
else
  launch_cmd=("$python_bin" -u train.py)
fi
launch_cmd+=("${common_args[@]}")

run_training_and_eval() {
  {
    echo "== Launch Command =="
    printf ' %q' "${launch_cmd[@]}"
    echo
    if "${launch_cmd[@]}"; then
      "$python_bin" auto_evaluate.py --model_type "$model_name" \
        --config_file "$model_log_dir/config.yaml" \
        --checkpoint_path "$model_log_dir/checkpoints/best.ckpt" \
        --output_dir "$model_log_dir/evaluation" --tag "$custom_tag"
    else
      train_exit_code=$?
      echo "Training failed with exit code $train_exit_code" >&2
      return "$train_exit_code"
    fi
  } 2>&1 | tee "$log_file"
}

if $background; then
  # Ensure log file directory exists
  mkdir -p "$(dirname "$log_file")"
  # Create empty log file to ensure it exists
  touch "$log_file"

  # When running in background, redirect all output to log file, don't print to terminal
  run_training_and_eval > /dev/null 2>&1 &
  bg_pid=$!

  # pid_file is already defined above, use path under model_log_dir
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

  # Only show startup info in terminal, don't show training output
  echo "Training started in background"
  echo "Log: $log_file"
  echo "PID: $bg_pid"
  echo "Settings recorded in: $pid_file"
  echo "View log: tail -f $log_file"
else
  run_training_and_eval
fi


# 
# bash train_ours.sh --model-name soon --lr 1e-3 --embed 256 --depth 7 --decoder-depth 1 --tag lr1e-3_embed256_depth7_dedep1