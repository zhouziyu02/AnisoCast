#!/usr/bin/env bash
# 超参数网格搜索脚本
# 搜索 learning_rate, embed_dim, depth, decoder_depth 四个参数

set -euo pipefail

self_name=$(basename "$0")
root_dir=$(cd "$(dirname "$0")" && pwd)
grid_dir="$root_dir/grid"
mkdir -p "$grid_dir"

# ==========================================
# 超参数候选值定义
# ==========================================
learning_rates=(5e-4 1e-3 2e-3 5e-3 1e-2)  # 5个候选值
embed_dims=(192 256 384 512 768)            # 5个候选值
depths=(6 8 10)                              # 3个候选值
decoder_depths=(1 2 3)                       # 3个候选值

# ==========================================
# 并发控制参数
# ==========================================
max_concurrent=5  # 最大并发进程数

# ==========================================
# 其他固定参数（可通过命令行覆盖）
# ==========================================
model_name="ost"
np=${NP:-4}  # 默认4卡，可通过环境变量NP覆盖
epochs=20
batch_size=32
weight_decay="1e-5"
drop_path=0.1
drop_rate=0.1
mlp_ratio=4.0
num_heads=16
patch_size=124
t_max=500
pred_len=2
grad_clip_norm=1.0
use_tensorboard=false
background=false  # 网格搜索时使用前台模式，便于管理

# 解析命令行参数（覆盖默认值）
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      model_name="$2"; shift 2 ;;
    --np)
      np="$2"; shift 2 ;;
    --epochs)
      epochs="$2"; shift 2 ;;
    --batch-size)
      batch_size="$2"; shift 2 ;;
    --max-concurrent)
      max_concurrent="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $self_name [options]"
      echo "Options:"
      echo "  --model-name <str>     模型名称 (默认: ost)"
      echo "  --np <int>              GPU数量 (默认: 4)"
      echo "  --epochs <int>          训练轮数 (默认: 20)"
      echo "  --batch-size <int>      批次大小 (默认: 32)"
      echo "  --max-concurrent <int>  最大并发数 (默认: 5)"
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1 ;;
  esac
done

# ==========================================
# 生成所有超参数组合
# ==========================================
generate_combinations() {
  local combinations=()
  for lr in "${learning_rates[@]}"; do
    for emb in "${embed_dims[@]}"; do
      for dep in "${depths[@]}"; do
        for dec in "${decoder_depths[@]}"; do
          # 生成setting字符串：lr{lr}_emb{emb}_dep{dep}_dec{dec}
          # 将科学计数法转换为下划线格式，例如 5e-4 -> 5e_4
          lr_str=$(echo "$lr" | sed 's/-/_/g')
          setting="lr${lr_str}_emb${emb}_dep${dep}_dec${dec}"
          combinations+=("$lr|$emb|$dep|$dec|$setting")
        done
      done
    done
  done
  printf '%s\n' "${combinations[@]}"
}

# ==========================================
# 运行单个训练任务
# ==========================================
run_single_task() {
  local lr="$1"
  local emb="$2"
  local dep="$3"
  local dec="$4"
  local setting="$5"
  
  local task_dir="$grid_dir/$setting"
  mkdir -p "$task_dir"
  
  local task_log="$task_dir/training.log"
  local task_pid="$task_dir/task.pid"
  
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 开始任务: $setting" | tee -a "$grid_dir/grid_search.log"
  echo "  参数: lr=$lr, embed_dim=$emb, depth=$dep, decoder_depth=$dec" | tee -a "$grid_dir/grid_search.log"
  
  # 记录任务开始时间（用于匹配结果文件）
  local task_start_time
  task_start_time=$(date +%s)
  
  # 记录任务开始信息
  {
    echo "setting=$setting"
    echo "learning_rate=$lr"
    echo "embed_dim=$emb"
    echo "depth=$dep"
    echo "decoder_depth=$dec"
    echo "start_time=$task_start_time"
    echo "pid=$$"
  } > "$task_pid"
  
  # 切换到根目录运行训练
  cd "$root_dir"
  
  # 生成tag（用于CSV文件名和匹配结果）
  local tag="${setting}"
  
  # 记录训练前的文件状态（用于后续匹配）
  local logs_before=$(ls -1 "$root_dir/logs/$model_name" 2>/dev/null | wc -l)
  local results_before=$(ls -1 "$root_dir/results/$model_name" 2>/dev/null | wc -l)
  local lightning_before=$(ls -1 "$root_dir/lightning_logs/version_*" 2>/dev/null | wc -l)
  
  # 运行训练（使用train_ours.sh，但重定向输出）
  local train_cmd=(
    bash train_ours.sh
    --model-name "$model_name"
    --lr "$lr"
    --embed "$emb"
    --depth "$dep"
    --decoder-depth "$dec"
    --epochs "$epochs"
    --batch "$batch_size"
    --weight-decay "$weight_decay"
    --drop-path "$drop_path"
    --drop-rate "$drop_rate"
    --mlp-ratio "$mlp_ratio"
    --heads "$num_heads"
    --patch "$patch_size"
    --t-max "$t_max"
    --pred-len "$pred_len"
    --grad-clip "$grad_clip_norm"
    --np "$np"
    --tag "$tag"
    --foreground
  )
  
  $use_tensorboard && train_cmd+=(--tensorboard)
  
  local train_exit_code=0
  
  # 运行训练，捕获输出
  if "${train_cmd[@]}" > "$task_log" 2>&1; then
    train_exit_code=0
  else
    train_exit_code=$?
  fi
  
  # 等待一小段时间，确保文件系统同步
  sleep 2
  
  # 训练完成后，查找并移动结果文件
  # 1. 查找logs目录中最新创建的文件（基于时间戳）
  local log_dirs=()
  while IFS= read -r -d '' dir; do
    local dir_mtime=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null)
    if [[ -n "$dir_mtime" && $dir_mtime -ge $task_start_time ]]; then
      log_dirs+=("$dir")
    fi
  done < <(find "$root_dir/logs/$model_name" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
  
  # 按修改时间排序，取最新的
  local latest_log_dir=""
  if [[ ${#log_dirs[@]} -gt 0 ]]; then
    latest_log_dir=$(printf '%s\n' "${log_dirs[@]}" | xargs -I{} sh -c 'echo "$(stat -c %Y {} 2>/dev/null || stat -f %m {} 2>/dev/null) {}"' | sort -rn | head -1 | cut -d' ' -f2-)
  fi
  
  if [[ -n "$latest_log_dir" && -d "$latest_log_dir" ]]; then
    # 复制配置文件、日志等到task_dir
    cp -r "$latest_log_dir"/* "$task_dir/" 2>/dev/null || true
  fi
  
  # 2. 查找lightning_logs中最新创建的version目录
  local version_dirs=()
  while IFS= read -r -d '' dir; do
    local dir_mtime=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null)
    if [[ -n "$dir_mtime" && $dir_mtime -ge $task_start_time ]]; then
      version_dirs+=("$dir")
    fi
  done < <(find "$root_dir/lightning_logs" -mindepth 1 -maxdepth 1 -type d -name "version_*" -print0 2>/dev/null)
  
  # 按修改时间排序，取最新的
  local latest_version=""
  if [[ ${#version_dirs[@]} -gt 0 ]]; then
    latest_version=$(printf '%s\n' "${version_dirs[@]}" | xargs -I{} sh -c 'echo "$(stat -c %Y {} 2>/dev/null || stat -f %m {} 2>/dev/null) {}"' | sort -rn | head -1 | cut -d' ' -f2-)
  fi
  
  if [[ -n "$latest_version" && -d "$latest_version" ]]; then
    # 复制checkpoint目录到task_dir
    if [[ -d "$latest_version/checkpoints" ]]; then
      mkdir -p "$task_dir/checkpoints"
      cp -r "$latest_version/checkpoints"/* "$task_dir/checkpoints/" 2>/dev/null || true
    fi
    # 复制config.yaml
    if [[ -f "$latest_version/config.yaml" ]]; then
      cp "$latest_version/config.yaml" "$task_dir/config.yaml" 2>/dev/null || true
    fi
  fi
  
  # 3. 查找评估结果CSV（在results目录，通过tag匹配）
  local results_dir="$root_dir/results/$model_name"
  if [[ -d "$results_dir" ]]; then
    # 查找包含tag的CSV文件，且创建时间在任务开始之后
    local csv_files=()
    while IFS= read -r -d '' file; do
      local file_mtime=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null)
      if [[ -n "$file_mtime" && $file_mtime -ge $task_start_time ]]; then
        if [[ "$file" == *"${tag}"* ]]; then
          csv_files+=("$file")
        fi
      fi
    done < <(find "$results_dir" -name "*.csv" -type f -print0 2>/dev/null)
    
    # 按修改时间排序，取最新的
    if [[ ${#csv_files[@]} -gt 0 ]]; then
      local latest_csv=$(printf '%s\n' "${csv_files[@]}" | xargs -I{} sh -c 'echo "$(stat -c %Y {} 2>/dev/null || stat -f %m {} 2>/dev/null) {}"' | sort -rn | head -1 | cut -d' ' -f2-)
      if [[ -n "$latest_csv" && -f "$latest_csv" ]]; then
        cp "$latest_csv" "$task_dir/metrics.csv" 2>/dev/null || true
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ 任务完成: $setting (CSV已保存)" | tee -a "$grid_dir/grid_search.log"
        
        # CSV保存成功后，删除checkpoint以节省空间
        if [[ -n "$latest_version" && -d "$latest_version" ]]; then
          if [[ -d "$latest_version/checkpoints" ]]; then
            rm -rf "$latest_version/checkpoints" 2>/dev/null || true
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] 🗑️  已删除checkpoint: $latest_version/checkpoints" | tee -a "$grid_dir/grid_search.log"
          fi
          # 可选：如果version目录为空，也可以删除整个目录
          # 但保留config.yaml可能有用，所以暂时只删除checkpoints
        fi
      fi
    else
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  任务完成: $setting (未找到匹配的CSV文件，保留checkpoint)" | tee -a "$grid_dir/grid_search.log"
    fi
  else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  任务完成: $setting (未找到results目录，保留checkpoint)" | tee -a "$grid_dir/grid_search.log"
  fi
  
  # 记录任务结束信息
  {
    echo "end_time=$(date +%s)"
    echo "exit_code=$train_exit_code"
    echo "status=$([ $train_exit_code -eq 0 ] && echo 'success' || echo 'failed')"
  } >> "$task_pid"
  
  return $train_exit_code
}

# ==========================================
# 进程池管理
# ==========================================
run_grid_search() {
  local combinations
  combinations=$(generate_combinations)
  local total_tasks
  total_tasks=$(echo "$combinations" | wc -l)
  
  echo "=========================================="
  echo "🔍 超参数网格搜索"
  echo "=========================================="
  echo "模型: $model_name"
  echo "总任务数: $total_tasks"
  echo "最大并发: $max_concurrent"
  echo "搜索参数:"
  echo "  learning_rate: ${learning_rates[*]}"
  echo "  embed_dim: ${embed_dims[*]}"
  echo "  depth: ${depths[*]}"
  echo "  decoder_depth: ${decoder_depths[*]}"
  echo "=========================================="
  echo ""
  
  # 初始化进度文件
  local progress_file="$grid_dir/progress.txt"
  local completed_file="$grid_dir/completed.txt"
  local failed_file="$grid_dir/failed.txt"
  > "$progress_file"
  > "$completed_file"
  > "$failed_file"
  
  local completed=0
  local failed=0
  local pids=()
  local tasks=()
  
  # 读取所有任务到数组
  mapfile -t tasks < <(echo "$combinations")
  
  # 处理每个任务
  local task_idx=0
  while [[ $task_idx -lt ${#tasks[@]} ]] || [[ ${#pids[@]} -gt 0 ]]; do
    # 启动新任务（如果还有未处理的任务且未达到最大并发数）
    while [[ ${#pids[@]} -lt $max_concurrent ]] && [[ $task_idx -lt ${#tasks[@]} ]]; do
      local task="${tasks[$task_idx]}"
      IFS='|' read -r lr emb dep dec setting <<< "$task"
      
      # 后台运行任务
      run_single_task "$lr" "$emb" "$dep" "$dec" "$setting" &
      local pid=$!
      pids+=("$pid")
      tasks[$task_idx]="$task|$pid"  # 保存pid用于追踪
      
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] 📌 启动任务 [$((task_idx+1))/$total_tasks]: $setting (PID: $pid)" | tee -a "$grid_dir/grid_search.log"
      ((task_idx++))
    done
    
    # 检查已完成的进程
    local new_pids=()
    local i=0
    while [[ $i -lt ${#pids[@]} ]]; do
      local pid="${pids[$i]}"
      if kill -0 "$pid" 2>/dev/null; then
        # 进程仍在运行
        new_pids+=("$pid")
      else
        # 进程已完成，检查退出码
        if wait "$pid" 2>/dev/null; then
          ((completed++))
          local task_info="${tasks[$((task_idx - ${#pids[@]} + i))]}"
          IFS='|' read -r _lr _emb _dep _dec _setting _ <<< "$task_info"
          echo "$_setting" >> "$completed_file"
          echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ 完成 [$completed/$total_tasks]: $_setting" | tee -a "$grid_dir/grid_search.log"
        else
          ((failed++))
          local task_info="${tasks[$((task_idx - ${#pids[@]} + i))]}"
          IFS='|' read -r _lr _emb _dep _dec _setting _ <<< "$task_info"
          echo "$_setting" >> "$failed_file"
          echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ 失败 [$failed/$total_tasks]: $_setting" | tee -a "$grid_dir/grid_search.log"
        fi
      fi
      ((i++))
    done
    pids=("${new_pids[@]}")
    
    # 更新进度
    echo "$completed/$total_tasks completed, $failed failed, ${#pids[@]} running" > "$progress_file"
    
    # 如果还有任务在运行，等待一小段时间
    if [[ ${#pids[@]} -gt 0 ]]; then
      sleep 5
    fi
  done
  
  echo ""
  echo "=========================================="
  echo "🎉 网格搜索完成！"
  echo "=========================================="
  echo "总任务数: $total_tasks"
  echo "成功: $completed"
  echo "失败: $failed"
  echo "结果目录: $grid_dir"
  echo "=========================================="
}

# ==========================================
# 主函数
# ==========================================
main() {
  # 创建grid目录和日志文件
  mkdir -p "$grid_dir"
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] 开始网格搜索" > "$grid_dir/grid_search.log"
  
  # 运行网格搜索
  run_grid_search
}

# 运行主函数
main "$@"

