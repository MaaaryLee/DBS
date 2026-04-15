# Results Guide

This folder contains the active outputs for the current quantization and deployment study.

## Start Here

If you only need the most important result files:

- [paper_quantization/quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)
  main evidence bundle and caveats
- [paper_quantization/pi_findings_since_96x96_threshold_2026-04-15.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/pi_findings_since_96x96_threshold_2026-04-15.md)
  plain-language PI summary
- [larger_models/summary_6d_models.json](/Users/maaary/Downloads/DBS-main/results/larger_models/summary_6d_models.json)
  6D size sweep summary across desktop TFLite and ESP32
- [esp32/repeats/native_fp32_96_96_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_summary.json)
  native ESP32 FP32 reference for the main candidate
- [esp32/repeats/native_int8_96_96_repaired_32k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json)
  native ESP32 INT8 result for the main candidate
- [larger_models/96_96/eval_tflite_96_96_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/eval_tflite_96_96_strict_holdout.json)
  strict held-out fidelity and replay-agreement evidence

## Folder Meanings

### `paper_quantization/`

Manuscript-facing summaries and assembled evidence bundles.

### `larger_models/`

Per-model export artifacts, desktop TFLite traces, and per-size evaluation files.

Important subfolders:

- `96_96/`
  main current deployment candidate
- `400_300/`
  larger scaling-study model inspired by prior work

### `esp32/`

On-device benchmark traces.

- `repeats/` contains the strongest reporting files because they summarize repeated runs
- top-level `espidf_*.json` files are older single-configuration traces from the architecture sweep

### `strict_eval/`

Strict calibration / held-out split metadata and replay episodes used to support the repaired fidelity and replay-based control analyses.

## How To Read The Results

Use this order:

1. `paper_quantization/`
   for the current narrative and main claims
2. `esp32/repeats/`
   for the deployment-relevant latency numbers
3. `larger_models/`
   for per-model artifacts and size-sweep context
4. `strict_eval/`
   for evidence that the held-out tests were truly disjoint from calibration

## Important Cautions

- Desktop TFLite is useful, but ESP32 is the deployment-relevant benchmark.
- `96x96` is the strongest current candidate, not a universal optimum.
- Cache configuration matters when interpreting FP32 vs INT8 on ESP32-S3.
- The larger `400x300` model is best treated as a scaling / memory-bottleneck study unless a stricter replication comparison is required.

## Archived Results

Legacy outputs and exploratory comparisons were moved to:

- [archive/repo_cleanup_2026-04-13/results_legacy](/Users/maaary/Downloads/DBS-main/archive/repo_cleanup_2026-04-13/results_legacy)

They are kept for reference, but they are not the main reporting path.
