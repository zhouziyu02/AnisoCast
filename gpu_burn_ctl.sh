#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/gpu_burn.pid"
LOG_FILE="${SCRIPT_DIR}/gpu_burn.log"
PY_SCRIPT="${SCRIPT_DIR}/gpu_burn_matmul.py"

usage() {
  cat <<EOF
Usage:
  $0 start [seconds] [size] [gpus]
  $0 stop
  $0 status

Examples:
  # 无限时间（直到 stop），size=32768，用所有 GPU
  $0 start

  # 跑 1800 秒，矩阵 size=32768，只用 0-7 号卡
  $0 start 1800 32768 0,1,2,3,4,5,6,7

  # 停止
  $0 stop

  # 查看状态
  $0 status
EOF
}

start() {
  local seconds="${1:-0}"      # 默认 0 => 无限
  local size="${2:-32768}"     # 默认矩阵大小
  local gpus="${3:-all}"       # 默认 all

  if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "Python script not found: ${PY_SCRIPT}"
    exit 1
  fi

  if [[ -f "${PID_FILE}" ]]; then
    local old_pid
    old_pid="$(cat "${PID_FILE}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "gpu_burn is already running with PID ${old_pid}."
      echo "Use '$0 stop' to stop it first."
      exit 1
    else
      rm -f "${PID_FILE}"
    fi
  fi

  echo "Starting gpu_burn in background..."
  echo "  seconds=${seconds}, size=${size}, gpus=${gpus}"
  echo "  log: ${LOG_FILE}"

  nohup python "${PY_SCRIPT}" \
    --seconds "${seconds}" \
    --size "${size}" \
    --gpus "${gpus}" \
    >"${LOG_FILE}" 2>&1 &

  echo $! > "${PID_FILE}"
  echo "Started with PID $(cat "${PID_FILE}")."
}

stop() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "No PID file found. Maybe not running?"
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Process with PID ${pid} is not running. Cleaning PID file."
    rm -f "${PID_FILE}"
    exit 0
  fi

  echo "Stopping gpu_burn (PID ${pid}) with SIGTERM..."
  kill "${pid}" || true
  sleep 3

  if kill -0 "${pid}" 2>/dev/null; then
    echo "Process still alive, sending SIGKILL..."
    kill -9 "${pid}" || true
    sleep 1
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    echo "Warning: process ${pid} still seems to be running."
  else
    echo "Stopped."
  fi

  rm -f "${PID_FILE}"
}

status() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "gpu_burn is not running (no PID file)."
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"

  if kill -0 "${pid}" 2>/dev/null; then
    echo "gpu_burn is RUNNING with PID ${pid}."
  else
    echo "gpu_burn PID file exists but process ${pid} is not running."
  fi
}

cmd="${1:-}"
case "${cmd}" in
  start)
    shift
    start "$@"
    ;;
  stop)
    stop
    ;;
  status)
    status
    ;;
  *)
    usage
    ;;
esac
