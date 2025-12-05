import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

model_path = Path('tflite_actors/model_int8.tflite')
if not model_path.exists():
    raise SystemExit('missing tflite_actors/model_int8.tflite')

states_path = Path('states_eval.npy')
if not states_path.exists():
    raise SystemExit('missing states_eval.npy')

delegate = None
delegate_candidates = [
    'tensorflowlite_xnnpack_delegate.dll',
    'libtensorflowlite_xnnpack_delegate.so',
    'libtensorflowlite_xnnpack_delegate.dylib',
]
for candidate in delegate_candidates:
    try:
        delegate = tf.lite.experimental.load_delegate(candidate)
        print(f'[info] Loaded delegate: {candidate}')
        break
    except (ValueError, OSError):
        continue

interpreter_kwargs = {'model_path': str(model_path)}
if delegate is not None:
    interpreter_kwargs['experimental_delegates'] = [delegate]

interpreter = tf.lite.Interpreter(**interpreter_kwargs)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']
locked_shape = [1] + list(input_details[0]['shape'][1:])
interpreter.resize_tensor_input(input_index, locked_shape)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

input_scale, input_zero_point = input_details[0]['quantization']
if input_scale == 0:
    raise SystemExit('input scale is zero; quantization info missing')

sample = np.load(states_path).astype('float32')[0:1]
quantized_sample = np.round(sample / input_scale + input_zero_point).astype(np.int8)
quantized_sample = quantized_sample.reshape(locked_shape)

for _ in range(10):
    interpreter.set_tensor(input_index, quantized_sample)
    interpreter.invoke()

runs = 500
latencies = []
for _ in range(runs):
    start = time.perf_counter()
    interpreter.set_tensor(input_index, quantized_sample)
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
output_path = Path('results/tflite_int8_latency.json')
output_path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
