import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import os

model_path = Path('tflite_actors/model_fp32.tflite')
if not model_path.exists():
    raise SystemExit('missing tflite_actors/model_fp32.tflite')

states_path = Path('states_eval.npy')

#
# Fair benchmarking note:
# - TF Lite may auto-enable CPU delegates (e.g., XNNPACK) for FP32 models.
# - To ensure apples-to-apples comparisons vs INT8, we disable *default delegates*
#   by default. Override with: $env:TFLITE_DISABLE_DEFAULT_DELEGATES="0"
#
disable_default_delegates = os.environ.get("TFLITE_DISABLE_DEFAULT_DELEGATES", "1") == "1"

interpreter_kwargs = {"model_path": str(model_path), "num_threads": 1}
if disable_default_delegates:
    try:
        interpreter_kwargs["experimental_op_resolver_type"] = (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
    except Exception:
        # Older TF builds may not expose this API; fall back to defaults.
        pass

try:
    interpreter = tf.lite.Interpreter(**interpreter_kwargs)
except TypeError:
    # Older TF builds may not accept experimental_op_resolver_type
    interpreter_kwargs.pop("experimental_op_resolver_type", None)
    interpreter = tf.lite.Interpreter(**interpreter_kwargs)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']
locked_shape = [1] + list(input_details[0]['shape'][1:])

# Only resize if needed (avoids warnings with delegates that require static tensors).
if list(input_details[0]['shape']) != locked_shape:
    interpreter.resize_tensor_input(input_index, locked_shape)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

obs_dim = int(locked_shape[1])
if not states_path.exists():
    # Synthetic fallback: simple Gaussian states with correct input dim.
    rng = np.random.default_rng(seed=0)
    states = rng.normal(loc=0.0, scale=1.0, size=(1000, obs_dim)).astype("float32")
    np.save(states_path, states)

states = np.load(states_path).astype('float32')
if states.ndim != 2 or states.shape[1] != obs_dim:
    # If an old states_eval.npy has incompatible shape (e.g., 6D), regenerate synthetic.
    rng = np.random.default_rng(seed=0)
    states = rng.normal(loc=0.0, scale=1.0, size=(1000, obs_dim)).astype("float32")
    np.save(states_path, states)

sample = states[0:1]
sample = sample.reshape(locked_shape)

for _ in range(10):
    interpreter.set_tensor(input_index, sample)
    interpreter.invoke()

runs = 500
latencies = []
for _ in range(runs):
    start = time.perf_counter()
    interpreter.set_tensor(input_index, sample)
    interpreter.invoke()
    latencies.append((time.perf_counter() - start) * 1000)

summary = {
    'model': str(model_path),
    'runs': runs,
    'mean_ms': float(np.mean(latencies)),
    'p50_ms': float(np.percentile(latencies, 50)),
    'p90_ms': float(np.percentile(latencies, 90)),
    'min_ms': float(np.min(latencies)),
    'max_ms': float(np.max(latencies)),
}

Path('results').mkdir(exist_ok=True)
output_path = Path('results/tflite_fp32_latency.json')
output_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
