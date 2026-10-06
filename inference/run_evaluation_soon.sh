#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$root_dir"
if [[ $# -lt 1 ]]; then
    echo 'Usage: bash inference/run_evaluation_soon.sh CHECKPOINT_PATH [additional evaluation arguments]' >&2
    exit 1
fi
checkpoint_path="$1"
shift
python3 -m inference.evaluate_soon --checkpoint_path "$checkpoint_path" "$@"
