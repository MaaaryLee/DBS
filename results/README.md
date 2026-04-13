# Results Guide

This folder contains the main benchmark and evaluation outputs used in the current quantization and deployment experiments.

## Start Here

- [paper_quantization/quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)
  Quantization metrics summary, including caveats.
- [paper_quantization/quantization_paper_metrics.json](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.json)
  Machine-readable version of the same summary.
- [larger_models/summary_6d_models.json](/Users/maaary/Downloads/DBS-main/results/larger_models/summary_6d_models.json)
  6D size-sweep summary across desktop TFLite and ESP32.

## Folder Meanings

### `benchmark_latest_tflite_only/`
- Desktop TFLite comparison traces.
- Most important file:
  - [benchmark_latest_tflite_only/benchmark_speedups.json](/Users/maaary/Downloads/DBS-main/results/benchmark_latest_tflite_only/benchmark_speedups.json)

### `esp32/`
- ESP32 benchmark traces.
- `repeats/` contains the repeated-run summaries that are strongest for reporting.
- Top-level `espidf_*.json` files are single configuration traces from the architecture sweep.

### `larger_models/`
- Per-model export artifacts and evaluation files for the 6D sweep.
- `96_96/` is the current main candidate folder.

### `strict_eval/`
- Strict calibration/held-out split artifacts and replay episodes.
- These files support the disjoint held-out fidelity test and replay-based control agreement analysis.

### `paper_quantization/`
- Quantization summary files for writing and review.

## Archived Results

Legacy 32x32-era outputs, temp comparisons, thread sweeps, and exploratory plots were moved to:

- [archive/repo_cleanup_2026-04-13/results_legacy](/Users/maaary/Downloads/DBS-main/archive/repo_cleanup_2026-04-13/results_legacy)

They are kept for reference.
