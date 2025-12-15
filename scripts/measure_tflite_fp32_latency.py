import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

model_path = Path('tflite_actors/model_fp32.tflite')
if not model_path.exists():
    raise SystemExit('missing tflite_actors/model_fp32.tflite')

states_path = Path('states_eval.npy')
if not states_path.exists():
    raise SystemExit('missing states_eval.npy')

#
# Fair benchmarking note:
# - We do NOT manually load delegates here.
# - TF Lite will pick its default delegate path consistently across scripts.
#
interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']
locked_shape = [1] + list(input_details[0]['shape'][1:])

# Only resize if needed (avoids warnings with delegates that require static tensors).
if list(input_details[0]['shape']) != locked_shape:
    interpreter.resize_tensor_input(input_index, locked_shape)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

sample = np.load(states_path).astype('float32')[0:1]
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
