#!/usr/bin/env python
import argparse
import time
import signal
import torch

STOP_REQUESTED = False


def handle_stop_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\n[Signal] Received signal {signum}, will stop after current step...", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple GPU burn script with large matmul.")
    parser.add_argument(
        "--seconds",
        type=int,
        default=0,
        help="How many seconds to run. <=0 means infinite (until stopped by signal)."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=32768,
        help="Matrix size N for NxN matmul (default: 32768). Increase if显存允许."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Data type for matmul (fp16 uses less memory, fp32更稳但更耗显存)."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help='GPU ids to use, e.g. "0,1,2,3" or "all" (default: all visible GPUs).'
    )
    parser.add_argument(
        "--no_grad",
        action="store_true",
        help="If set,只做前向 matmul 不做 backward."
    )
    return parser.parse_args()


def get_devices(gpu_arg: str):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")

    n = torch.cuda.device_count()
    if gpu_arg == "all":
        return list(range(n))

    ids = []
    for x in gpu_arg.split(","):
        x = x.strip()
        if not x:
            continue
        idx = int(x)
        if idx < 0 or idx >= n:
            raise ValueError(f"Invalid GPU id {idx}, available: 0..{n-1}")
        ids.append(idx)
    return ids


def main():
    # 注册信号处理，支持 kill 优雅退出
    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    args = parse_args()
    devices = get_devices(args.gpus)

    print(f"Using GPUs: {devices}")
    print(f"Matrix size: {args.size} x {args.size}")
    print(f"Run seconds: {args.seconds if args.seconds > 0 else 'INF (until stopped)'}")
    print(f"Data type: {args.dtype}")
    print(f"Backward: {'OFF (no_grad)' if args.no_grad else 'ON (with backward)'}")
    print("Press Ctrl+C (foreground) or send SIGTERM (kill) to stop.\n", flush=True)

    if args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 预先在每张卡上分配矩阵
    mats = {}
    for dev in devices:
        device = torch.device(f"cuda:{dev}")
        print(f"Allocating matrices on cuda:{dev} ...", flush=True)
        a = torch.randn(
            args.size, args.size, device=device, dtype=dtype,
            requires_grad=not args.no_grad,
        )
        b = torch.randn(
            args.size, args.size, device=device, dtype=dtype,
            requires_grad=not args.no_grad,
        )
        mats[dev] = (a, b)

    start_time = time.time()

    try:
        step = 0
        while True:
            if STOP_REQUESTED:
                print("Stop requested, exiting loop...", flush=True)
                break

            if args.seconds > 0 and (time.time() - start_time) > args.seconds:
                print("Time limit reached, exiting loop...", flush=True)
                break

            step += 1
            # 每个 GPU 上做一次 matmul
            for dev in devices:
                a, b = mats[dev]
                device = torch.device(f"cuda:{dev}")
                with torch.cuda.device(device):
                    c = torch.matmul(a, b)
                    if not args.no_grad:
                        loss = c.sum()
                        loss.backward()
                    else:
                        _ = c.sum()

            # 同步确保计算完成
            for dev in devices:
                torch.cuda.synchronize(dev)

            if step % 5 == 0:
                elapsed = time.time() - start_time
                print(f"[Step {step}] elapsed = {elapsed:.1f}s", flush=True)

    except Exception as e:
        print(f"[Error] {e}", flush=True)

    finally:
        del mats
        torch.cuda.empty_cache()
        print("Cleanup done. Bye.", flush=True)


if __name__ == "__main__":
    main()
