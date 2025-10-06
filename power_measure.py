#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import os
import json
import numpy as np
import torch
from stable_baselines3 import TD3
from BGN_MC_fixed import BGN_MC_Fixed

# Minimal, CPU-only batch inference workload over saved states

def run_inference(policy, states: np.ndarray, duration_s: int = 180) -> None:
    policy.eval()
    device = torch.device('cpu')
    torch.set_num_threads(os.cpu_count() or 4)
    x = torch.from_numpy(states).float().to(device)
    # Loop for duration_s seconds
    t0 = time.time()
    with torch.no_grad():
        while time.time() - t0 < duration_s:
            _ = policy(x)


def launch_powermetrics(outfile: str, interval_ms: int = 100) -> subprocess.Popen:
    cmd = [
        'sudo','-n','powermetrics',
        '--samplers','cpu_power',
        '--hide-cpu-duty-cycle',
        '-i', str(interval_ms),
    ]
    f = open(outfile, 'w')
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


def _extract_power(line: str, label: str):
    if label not in line:
        return None
    # handle both "W" and "mW"
    try:
        seg = line.split(label)[1].strip()
        # e.g., "3147 mW" or "3.15 W"
        if seg.endswith('mW'):
            val = float(seg[:-2].strip()) / 1000.0
        elif seg.endswith('W'):
            val = float(seg[:-1].strip())
        else:
            # try splitting by space and take first token
            tok = seg.split()[0]
            val = float(tok)
    except Exception:
        return None
    return val


def parse_powermetrics_log(path: str) -> dict:
    # Prefer Combined Power (CPU + GPU + ANE) if present; otherwise use CPU Power
    if not os.path.exists(path):
        return { 'samples': 0, 'mean_W': None, 'std_W': None }
    cpu_powers = []
    combo_powers = []
    with open(path, 'r', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if 'Combined Power' in line:
                val = _extract_power(line, 'Combined Power (CPU + GPU + ANE):')
                if val is None:
                    # Some builds format as "Combined Power: X mW"
                    val = _extract_power(line, 'Combined Power:')
                if val is not None:
                    combo_powers.append(val)
            elif 'CPU Power:' in line:
                val = _extract_power(line, 'CPU Power:')
                if val is not None:
                    cpu_powers.append(val)
    powers = combo_powers if combo_powers else cpu_powers
    if not powers:
        return { 'samples': 0, 'mean_W': None, 'std_W': None }
    arr = np.asarray(powers, dtype=float)
    return {
        'samples': int(arr.size),
        'mean_W': float(arr.mean()),
        'std_W': float(arr.std(ddof=0)),
        'min_W': float(arr.min()),
        'p50_W': float(np.percentile(arr, 50)),
        'p90_W': float(np.percentile(arr, 90)),
        'max_W': float(arr.max()),
        'source': 'Combined' if combo_powers else 'CPU'
    }


def load_policies(model_zip='models/TD3_64_64/1500.zip'):
    env = BGN_MC_Fixed(tmax=1100, pd=True)
    model = TD3.load(model_zip, env=env)
    fp32_policy = model.policy.actor.to(torch.device('cpu'))
    fp32_policy.eval()
    # Build INT8 weights-only version from our working script
    from real_quantization_working import quantize_model_real
    int8_policy = quantize_model_real(fp32_policy)
    int8_policy.eval()
    return fp32_policy, int8_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fp32','int8'], required=True)
    parser.add_argument('--duration', type=int, default=180)
    parser.add_argument('--states', type=str, default='states_eval.npy')
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--interval_ms', type=int, default=100)
    parser.add_argument('--no-pm', action='store_true', help='Do not launch powermetrics (external capture).')
    args = parser.parse_args()

    if args.log is None:
        args.log = f'powermetrics_{args.mode}.log'

    if not os.path.exists(args.states):
        print(f"States file not found: {args.states}", file=sys.stderr)
        sys.exit(2)

    states = np.load(args.states)

    fp32_policy, int8_policy = load_policies()
    policy = fp32_policy if args.mode == 'fp32' else int8_policy

    pm = None
    if not args.no_pm:
        try:
            pm = launch_powermetrics(args.log, args.interval_ms)
        except Exception as e:
            print('Failed to start powermetrics without sudo cache. Please run this once in another terminal:', file=sys.stderr)
            print('  sudo -v', file=sys.stderr)
            sys.exit(1)

    try:
        run_inference(policy, states, args.duration)
    finally:
        if pm is not None:
            pm.terminate()
            try:
                pm.wait(timeout=5)
            except Exception:
                pm.kill()

    stats = parse_powermetrics_log(args.log)
    print(json.dumps({ 'mode': args.mode, 'duration_s': args.duration, 'interval_ms': args.interval_ms, 'stats': stats }, indent=2))

if __name__ == '__main__':
    main()
