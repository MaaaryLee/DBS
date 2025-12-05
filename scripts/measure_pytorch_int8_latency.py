import torch
import time
import json
import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quantize_model

sys.modules['__main__'].TD3Actor = quantize_model.TD3Actor
sys.modules['__main__'].QuantizableTD3Actor = quantize_model.QuantizableTD3Actor

actor_path = pathlib.Path('models/policies/actor_int8_static_32_32.pt')
states_path = pathlib.Path('states_eval.npy')

if not actor_path.exists():
    raise SystemExit('missing actor_int8_static_32_32.pt')
if not states_path.exists():
    raise SystemExit('missing states_eval.npy')

actor = torch.load(actor_path, map_location='cpu')
actor.eval()

sample = np.load(states_path).astype('float32')[0:1]
sample = torch.from_numpy(sample)

runs = 500

with torch.inference_mode():
    for _ in range(10):
        actor(sample)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        actor(sample)
        times.append((time.perf_counter() - start) * 1000)

summary = {
    'model': 'actor_int8_static_32_32_pytorch_full',
    'runs': runs,
    'mean_ms': float(np.mean(times)),
    'p50_ms': float(np.percentile(times, 50)),
    'p90_ms': float(np.percentile(times, 90)),
    'min_ms': float(np.min(times)),
    'max_ms': float(np.max(times)),
}

results_dir = pathlib.Path('results')
results_dir.mkdir(exist_ok=True)
output_path = results_dir / 'pytorch_int8_latency.json'
output_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
