#!/usr/bin/env bash
set -euo pipefail

# Run one batch at a time by default; control total concurrency explicitly.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
max_jobs="${BATCH_JOBS:-1}"
workers="${WORKERS:-4}"
python_bin="${PYTHON:-python}"
[[ "$max_jobs" =~ ^[1-9][0-9]*$ && "$workers" =~ ^[1-9][0-9]*$ ]] || {
  echo "BATCH_JOBS and WORKERS must be positive integers" >&2; exit 2;
}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
log_dir="${LOG_DIR:-${script_dir}/../logs/data_pressure}"
mkdir -p "$log_dir"
pids=()
failed=0
for start_year in 1979 1984 1989 1994 1999 2004 2009 2014; do
  end_year=$((start_year + 4))
  "$python_bin" "${script_dir}/parallel_download_pressure_1p5.py" \
    --start "${start_year}-01-01" --end "${end_year}-12-31" \
    --workers "$workers" --anon > "${log_dir}/${start_year}_${end_year}.log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} >= max_jobs )); then
    if wait "${pids[0]}"; then :; else failed=1; fi
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  if wait "$pid"; then :; else failed=1; fi
done
echo "Download batches finished; failures=${failed}; logs=${log_dir}"
exit "$failed"
