#!/usr/bin/env bash
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET

set -euo pipefail

self_name=$(basename "$0")
root_dir=$(cd "$(dirname "$0")" && pwd)
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
data_dir='./data/S2S'  # Update this path to your data directory
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
  --embed, --embed-dim <int>        Embedding dimension (must be divisible by number of heads)
  --patch, --patch-size <int>       Patch size
  --depth <int>                     Number of Transformer layers
  --decoder-depth <int>             Decoder depth
  --heads, --num-heads <int>        Number of attention heads
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

common_args=(--config_filepath "$config_file" --devices "$np" --accelerator gpu)
if (( np > 1 )); then
  common_args+=(--strategy ddp_find_unused_parameters_true)
fi
$use_tensorboard && common_args+=(--use_tensorboard)

if (( np > 1 )); then
  # Explicitly specify master_port to avoid port conflicts in multi-task scenarios
  launch_cmd=(torchrun --standalone --nproc_per_node="$np" --master_port="$MASTER_PORT" train.py)
else
  launch_cmd=(python3 -u train.py)
fi
launch_cmd+=("${common_args[@]}")

find_checkpoint_by_config() {
  local config_path="$1"
  local start_time="$2"
  local expected_model_name="$3"  # Expected model name

  # Calculate hash of current config file (for matching)
  local config_hash
  config_hash=$(md5sum "$config_path" 2>/dev/null | cut -d' ' -f1 || md5 -q "$config_path" 2>/dev/null)

  # Extract model_name from config file (for double verification)
  local config_model_name
  config_model_name=$(grep -E "^model_name:" "$config_path" 2>/dev/null | head -1 | sed -E "s/.*model_name:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/" | tr -d ' ')

  # Find all version directories in lightning_logs (sorted by time, newest first)
  local version_dirs=()
  mapfile -t version_dirs < <(find lightning_logs -maxdepth 1 -type d -name "version_*" -printf '%T@ %p\n' 2>/dev/null | \
                        sort -rn | cut -d' ' -f2- | head -20)

  # If find doesn't support -printf, use ls
  if [[ ${#version_dirs[@]} -eq 0 ]]; then
    mapfile -t version_dirs < <(ls -td lightning_logs/version_* 2>/dev/null | head -20)
  fi

  # Method 1: Match by config.yaml content + model_name verification + timestamp verification (most reliable)
  for version_dir in "${version_dirs[@]}"; do
    local version_config="$version_dir/config.yaml"
    if [[ -f "$version_config" ]]; then
      local version_hash
      version_hash=$(md5sum "$version_config" 2>/dev/null | cut -d' ' -f1 || md5 -q "$version_config" 2>/dev/null)

        # If config file content is the same
        if [[ "$config_hash" == "$version_hash" ]]; then
          # Verify if model_name in version directory matches (double verification)
          local version_model_name
          version_model_name=$(grep -E "^model_name:" "$version_config" 2>/dev/null | head -1 | sed -E "s/.*model_name:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/" | tr -d ' ')

          # If expected model_name is provided, it must match
          if [[ -n "$expected_model_name" && -n "$version_model_name" ]]; then
            if [[ "$expected_model_name" != "$version_model_name" ]]; then
              continue  # model_name doesn't match, skip this version directory
            fi
          fi

          # Verify timestamp: config.yaml creation time should be after training start (allow 5 min error)
          local version_mtime
          version_mtime=$(stat -c %Y "$version_config" 2>/dev/null || stat -f %m "$version_config" 2>/dev/null)
          if [[ -n "$version_mtime" && $version_mtime -lt $((start_time - 300)) ]]; then
            continue  # Timestamp doesn't match, skip
          fi

          # Find checkpoint in this directory
          # Prioritize finding best checkpoint (exclude last.ckpt, as we only save best model now)
          local best_checkpoint
          best_checkpoint=$(find "$version_dir/checkpoints" -name "epoch=*-step=*.ckpt" -type f 2>/dev/null | sort -r | head -1)
          if [[ -n "$best_checkpoint" && -f "$best_checkpoint" ]]; then
            # Verify checkpoint modification time should be after training start
          local ckpt_mtime
          ckpt_mtime=$(stat -c %Y "$best_checkpoint" 2>/dev/null || stat -f %m "$best_checkpoint" 2>/dev/null)
          if [[ -n "$ckpt_mtime" && $ckpt_mtime -ge $start_time ]]; then
            echo "$best_checkpoint"
            return 0
          fi
        fi

          # If epoch=*-step=*.ckpt format not found, fallback to finding all .ckpt files
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

  # Method 2: Match by timestamp + model_name (if config content matching fails)
  for version_dir in "${version_dirs[@]}"; do
    local version_config="$version_dir/config.yaml"
    if [[ -f "$version_config" ]]; then
      local version_mtime
      version_mtime=$(stat -c %Y "$version_config" 2>/dev/null || stat -f %m "$version_config" 2>/dev/null)

        # If version's config.yaml creation time is after training start (allow 5 min error)
        if [[ -n "$version_mtime" && $version_mtime -ge $((start_time - 300)) ]]; then
          # Verify if model_name matches
          if [[ -n "$expected_model_name" ]]; then
            local version_model_name
            version_model_name=$(grep -E "^model_name:" "$version_config" 2>/dev/null | head -1 | sed -E "s/.*model_name:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/" | tr -d ' ')
            if [[ -n "$version_model_name" && "$expected_model_name" != "$version_model_name" ]]; then
              continue  # model_name doesn't match, skip
            fi
          fi

          # Prioritize finding best checkpoint (exclude last.ckpt)
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

          # If epoch=*-step=*.ckpt format not found, fallback to finding all .ckpt files
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
  local checkpoint_version_dir=""  # Store checkpoint version directory path
  local checkpoint_info_file="$config_file.checkpoint_dir"

  # Clean up any existing old files
  rm -f "$checkpoint_info_file"

  {
    echo "== Launch Command =="
    printf ' %q' "${launch_cmd[@]}"
    echo
    echo

    # Run training command while capturing checkpoint path
    "${launch_cmd[@]}" 2>&1 | tee >(while IFS= read -r line; do
      echo "$line"
      # Extract checkpoint directory path from training output
      if [[ "$line" =~ Config.*saved.*checkpoint.*directory:\ ([^[:space:]]+) ]]; then
        local extracted_path="${BASH_REMATCH[1]}"
        # If it's config.yaml path, convert to version directory
        extracted_path="${extracted_path%/config.yaml}"
        echo "$extracted_path" > "$checkpoint_info_file"
      fi
    done)

    train_exit_code=${PIPESTATUS[0]}
    echo
    echo "Training exit code: $train_exit_code"

    # Read saved checkpoint directory path
    if [[ -f "$checkpoint_info_file" ]]; then
      checkpoint_version_dir=$(cat "$checkpoint_info_file" 2>/dev/null)
      rm -f "$checkpoint_info_file"
    fi

    if [[ $train_exit_code -eq 0 ]]; then
      echo "Searching for checkpoint corresponding to current training task..."
      echo "Expected model name: $model_name"

      local found_checkpoint=""

      # Method 1: If checkpoint directory extracted from training log, use it directly
      if [[ -n "$checkpoint_version_dir" && -d "$checkpoint_version_dir" ]]; then
        echo "Using checkpoint directory saved during training: $checkpoint_version_dir"
        # Find best checkpoint in this directory
        found_checkpoint=$(find "$checkpoint_version_dir/checkpoints" -name "epoch=*-step=*.ckpt" -type f 2>/dev/null | sort -r | head -1)
        if [[ -z "$found_checkpoint" || ! -f "$found_checkpoint" ]]; then
          # If epoch=*-step=*.ckpt format not found, find all .ckpt files
          found_checkpoint=$(find "$checkpoint_version_dir/checkpoints" -name "*.ckpt" -type f 2>/dev/null | sort -r | head -1)
        fi
      fi

      # Method 2: If method 1 fails, use original search logic
      if [[ -z "$found_checkpoint" || ! -f "$found_checkpoint" ]]; then
        echo "Checkpoint path not extracted from training log, using auto-search mode..."
        found_checkpoint=$(find_checkpoint_by_config "$config_file" "$training_start_time" "$model_name")
      fi

      if [[ -n "$found_checkpoint" && -f "$found_checkpoint" ]]; then
        echo "Found checkpoint: $found_checkpoint"
        echo "Starting automatic evaluation of $model_name model..."
        if [[ -n "$custom_tag" ]]; then
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file" --checkpoint_path "$found_checkpoint" --tag "$custom_tag"
        else
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file" --checkpoint_path "$found_checkpoint"
        fi
        return $?
      else
        echo "Corresponding checkpoint not found, using auto-search mode..."
        if [[ -n "$custom_tag" ]]; then
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file" --tag "$custom_tag"
        else
          python3 auto_evaluate.py --model_type "$model_name" --config_file "$config_file"
        fi
        return $?
      fi
    else
        echo "Training failed"
      return $train_exit_code
    fi
  } | tee "$log_file"

  return ${PIPESTATUS[0]}
}

if $background; then
  # Ensure log file directory exists
  mkdir -p "$(dirname "$log_file")"
  # Create empty log file to ensure it exists
  touch "$log_file"

  # When running in background, redirect all output to log file, don't print to terminal
  run_training_and_eval > "$log_file" 2>&1 &
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