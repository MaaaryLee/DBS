import json
import pathlib
import matplotlib.pyplot as plt

results_dir = pathlib.Path('results')
fp32_path = results_dir / 'fp32_latency.json'
pytorch_int8_path = results_dir / 'pytorch_int8_latency.json'
tflite_fp32_path = results_dir / 'tflite_fp32_latency.json'
tflite_int8_path = results_dir / 'tflite_int8_latency.json'

missing = [p for p in [fp32_path, pytorch_int8_path, tflite_fp32_path, tflite_int8_path] if not p.exists()]
if missing:
    raise SystemExit(f"Missing latency JSON files: {', '.join(str(m) for m in missing)}")

with fp32_path.open() as f:
    fp32_stats = json.load(f)
with pytorch_int8_path.open() as f:
    pytorch_int8_stats = json.load(f)
with tflite_fp32_path.open() as f:
    tflite_fp32_stats = json.load(f)
with tflite_int8_path.open() as f:
    tflite_int8_stats = json.load(f)

labels = ['FP32 (PyTorch)', 'INT8 (PyTorch)', 'FP32 (TFLite)', 'INT8 (TFLite)']
means = [
    fp32_stats['mean_ms'],
    pytorch_int8_stats['mean_ms'],
    tflite_fp32_stats['mean_ms'],
    tflite_int8_stats['mean_ms'],
]
colors = ['#3498db', '#9b59b6', '#1abc9c', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, means, color=colors)
ax.set_ylabel('Mean Inference Time (ms)')
ax.set_title('PyTorch vs TFLite Latency Comparison')
ax.grid(axis='y', alpha=0.3)

for bar, value in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, value,
            f'{value:.4f} ms', ha='center', va='bottom')

plt.tight_layout()
output_path = results_dir / 'latency_comparison_pytorch_vs_tflite.png'
fig.savefig(output_path, dpi=150)
plt.close(fig)
print(f'[OK] Saved {output_path}')
