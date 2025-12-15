import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.quantize_model as quantize_model

sys.modules['__main__'].TD3Actor = quantize_model.TD3Actor
sys.modules['__main__'].QuantizableTD3Actor = quantize_model.QuantizableTD3Actor

parser = argparse.ArgumentParser(description="Measure PyTorch FP32 actor latency (default: 22x22).")
parser.add_argument("--h1", type=int, default=22)
parser.add_argument("--h2", type=int, default=22)
parser.add_argument("--states-path", type=str, default="states_eval.npy")
parser.add_argument("--runs", type=int, default=500)
parser.add_argument("--warmup", type=int, default=25)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--threads", type=int, default=1, help="torch.set_num_threads(...)")
parser.add_argument("--output", type=str, default="results/fp32_latency.json")
args = parser.parse_args()

actor_path = pathlib.Path(f"models/policies/actor_fp32_{args.h1}_{args.h2}.pt")
states_path = pathlib.Path(args.states_path)

if not actor_path.exists():
    raise SystemExit(f"missing {actor_path}")
if not states_path.exists():
    raise SystemExit(f"missing {states_path}")

actor = torch.load(actor_path, map_location='cpu')
actor.eval()

torch.set_num_threads(args.threads)
try:
    torch.set_num_interop_threads(args.threads)
except Exception:
    pass

states = np.load(states_path).astype('float32')
batch = max(1, int(args.batch_size))
sample = torch.from_numpy(states[:batch])

runs = args.runs
warmup = args.warmup

with torch.inference_mode():
    for _ in range(warmup):
        actor(sample)
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        actor(sample)
        times.append((time.perf_counter_ns() - start) / 1_000_000.0)

summary = {
    'model': f'actor_fp32_{args.h1}_{args.h2}',
    'runs': runs,
    'warmup': warmup,
    'batch_size': batch,
    'threads': args.threads,
    'checkpoint': str(actor_path),
    'checkpoint_size_kb': float(actor_path.stat().st_size / 1024.0),
    'mean_ms': float(np.mean(times)),
    'std_ms': float(np.std(times)),
    'p50_ms': float(np.percentile(times, 50)),
    'p90_ms': float(np.percentile(times, 90)),
    'min_ms': float(np.min(times)),
    'max_ms': float(np.max(times)),
}

output_path = pathlib.Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
