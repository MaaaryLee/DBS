# Repository Organization

This file is a short map of the folders that matter most for the current training, quantization, and deployment workflow.

## Read These First

- [README.md](/Users/maaary/Downloads/DBS-main/README.md)
  project overview and main workflows
- [results/README.md](/Users/maaary/Downloads/DBS-main/results/README.md)
  guide to the result files worth citing
- [quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)
  main evidence bundle
- [pi_findings_since_96x96_threshold_2026-04-15.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/pi_findings_since_96x96_threshold_2026-04-15.md)
  PI-facing summary of the recent research process

## Active Research Folders

### `core/`

Primary research code.

Most important files:

- [training.py](/Users/maaary/Downloads/DBS-main/core/training.py)
- [BGN_MC.py](/Users/maaary/Downloads/DBS-main/core/BGN_MC.py)
- [BGN_MC_Online.py](/Users/maaary/Downloads/DBS-main/core/BGN_MC_Online.py)
- [quantize_model.py](/Users/maaary/Downloads/DBS-main/core/quantize_model.py)

### `deployment/`

Model export pipeline.

Most important files:

- [convert_to_onnx.py](/Users/maaary/Downloads/DBS-main/deployment/convert_to_onnx.py)
- [convert_onnx_to_tf.py](/Users/maaary/Downloads/DBS-main/deployment/convert_onnx_to_tf.py)
- [convert_tf_to_tflite.py](/Users/maaary/Downloads/DBS-main/deployment/convert_tf_to_tflite.py)
- [convert_tflite_to_c.py](/Users/maaary/Downloads/DBS-main/deployment/convert_tflite_to_c.py)

### `scripts/`

Reproducible helpers for benchmarking, evaluation, and firmware preparation.

Most important files:

- [build_model_family_artifacts.py](/Users/maaary/Downloads/DBS-main/scripts/build_model_family_artifacts.py)
- [convert_saved_model_to_tflite_int8.py](/Users/maaary/Downloads/DBS-main/scripts/convert_saved_model_to_tflite_int8.py)
- [evaluate_tflite_quantization.py](/Users/maaary/Downloads/DBS-main/scripts/evaluate_tflite_quantization.py)
- [measure_tflite_fp32_latency.py](/Users/maaary/Downloads/DBS-main/scripts/measure_tflite_fp32_latency.py)
- [measure_tflite_int8_latency.py](/Users/maaary/Downloads/DBS-main/scripts/measure_tflite_int8_latency.py)
- [prepare_esp32_benchmark.py](/Users/maaary/Downloads/DBS-main/scripts/prepare_esp32_benchmark.py)
- [run_esp32_benchmark.py](/Users/maaary/Downloads/DBS-main/scripts/run_esp32_benchmark.py)
- [run_espidf_benchmark_variant.py](/Users/maaary/Downloads/DBS-main/scripts/run_espidf_benchmark_variant.py)

### `esp32_firmware/`

Arduino-facing firmware.

Most important files:

- [README.md](/Users/maaary/Downloads/DBS-main/esp32_firmware/README.md)
- [dbs_inference.ino](/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_inference.ino)
- [dbs_benchmark.ino](/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_benchmark/dbs_benchmark.ino)
- [model.h](/Users/maaary/Downloads/DBS-main/esp32_firmware/model.h)

### `espidf_firmware/`

Native ESP32 benchmark project for the paper-faithful runtime path:

- [dbs_espnn_benchmark](/Users/maaary/Downloads/DBS-main/espidf_firmware/dbs_espnn_benchmark)

### `models/`

Training checkpoints and exported policy artifacts.

Most relevant folders right now:

- [TD3_80_80](/Users/maaary/Downloads/DBS-main/models/TD3_80_80)
- [TD3_96_96](/Users/maaary/Downloads/DBS-main/models/TD3_96_96)
- [TD3_128_128](/Users/maaary/Downloads/DBS-main/models/TD3_128_128)
- [TD3_400_300](/Users/maaary/Downloads/DBS-main/models/TD3_400_300)
- [policies](/Users/maaary/Downloads/DBS-main/models/policies)

### `results/`

Active benchmark and evaluation outputs used for review and writing.

See:

- [results/README.md](/Users/maaary/Downloads/DBS-main/results/README.md)

## Supporting Data Folders

### `matlab/`

MATLAB simulator and cached source data.

### `matlab_data/`

MATLAB Online bridge artifacts and downloaded simulation outputs.

## Root-Level Files Worth Noticing

- [states_eval_4d.npy](/Users/maaary/Downloads/DBS-main/states_eval_4d.npy)
  legacy 4D state bank
- [states_eval_6d.npy](/Users/maaary/Downloads/DBS-main/states_eval_6d.npy)
  original degenerate 6D file retained for traceability
- [states_eval_6d_repaired.npy](/Users/maaary/Downloads/DBS-main/states_eval_6d_repaired.npy)
  repaired 6D state bank used in later evaluation work
- [model.h](/Users/maaary/Downloads/DBS-main/model.h)
  mirrored active model header for older flows that still expect a root-level header

## Archived Material

Legacy and exploratory material was moved to:

- [archive/repo_cleanup_2026-04-13](/Users/maaary/Downloads/DBS-main/archive/repo_cleanup_2026-04-13)

These files are kept for traceability, but they are not part of the main workflow.
