# Repository Organization

This file lists the main project folders and the files that matter most for training, deployment, and benchmarking.

## Key Files

- [README.md](/Users/maaary/Downloads/DBS-main/README.md)
  Project overview, workflow, and deployment notes.
- [results/paper_quantization/quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)
  Quantization metrics summary and caveats.
- [results/larger_models/summary_6d_models.json](/Users/maaary/Downloads/DBS-main/results/larger_models/summary_6d_models.json)
  Size sweep summary for the 6D TFLite and ESP32 experiments.
- [results/README.md](/Users/maaary/Downloads/DBS-main/results/README.md)
  Guide to the main result files.

## Active Folders

### `core/`
- Primary research code.
- Most important files:
  - [core/training.py](/Users/maaary/Downloads/DBS-main/core/training.py)
  - [core/BGN_MC.py](/Users/maaary/Downloads/DBS-main/core/BGN_MC.py)
  - [core/BGN_MC_Online.py](/Users/maaary/Downloads/DBS-main/core/BGN_MC_Online.py)
  - [core/quantize_model.py](/Users/maaary/Downloads/DBS-main/core/quantize_model.py)

### `deployment/`
- Export pipeline from trained policy to deployable model formats.
- Most important files:
  - [deployment/convert_to_onnx.py](/Users/maaary/Downloads/DBS-main/deployment/convert_to_onnx.py)
  - [deployment/convert_onnx_to_tf.py](/Users/maaary/Downloads/DBS-main/deployment/convert_onnx_to_tf.py)
  - [deployment/convert_tf_to_tflite.py](/Users/maaary/Downloads/DBS-main/deployment/convert_tf_to_tflite.py)
  - [deployment/convert_tflite_to_c.py](/Users/maaary/Downloads/DBS-main/deployment/convert_tflite_to_c.py)

### `scripts/`
- Reproducible experiment helpers and benchmark runners.
- Most important files:
  - [scripts/convert_saved_model_to_tflite_int8.py](/Users/maaary/Downloads/DBS-main/scripts/convert_saved_model_to_tflite_int8.py)
  - [scripts/evaluate_tflite_quantization.py](/Users/maaary/Downloads/DBS-main/scripts/evaluate_tflite_quantization.py)
  - [scripts/measure_tflite_fp32_latency.py](/Users/maaary/Downloads/DBS-main/scripts/measure_tflite_fp32_latency.py)
  - [scripts/measure_tflite_int8_latency.py](/Users/maaary/Downloads/DBS-main/scripts/measure_tflite_int8_latency.py)
  - [scripts/prepare_esp32_benchmark.py](/Users/maaary/Downloads/DBS-main/scripts/prepare_esp32_benchmark.py)
  - [scripts/run_esp32_benchmark.py](/Users/maaary/Downloads/DBS-main/scripts/run_esp32_benchmark.py)
  - [scripts/run_espidf_benchmark_variant.py](/Users/maaary/Downloads/DBS-main/scripts/run_espidf_benchmark_variant.py)
  - [scripts/assemble_quantization_paper_metrics.py](/Users/maaary/Downloads/DBS-main/scripts/assemble_quantization_paper_metrics.py)

### `matlab/`
- MATLAB BGN simulator and cached source data.
- Most important files:
  - [matlab/bgn_init.m](/Users/maaary/Downloads/DBS-main/matlab/bgn_init.m)
  - [matlab/bgn_step.m](/Users/maaary/Downloads/DBS-main/matlab/bgn_step.m)
  - [matlab/bgn_vars.mat](/Users/maaary/Downloads/DBS-main/matlab/bgn_vars.mat)

### `matlab_data/`
- MATLAB Online bridge artifacts and downloaded simulation outputs.
- Most important files:
  - [matlab_data/run_simulation_online.m](/Users/maaary/Downloads/DBS-main/matlab_data/run_simulation_online.m)
  - [matlab_data/simulation_results.mat](/Users/maaary/Downloads/DBS-main/matlab_data/simulation_results.mat)
  - [matlab_data/simulation_results-2.mat](/Users/maaary/Downloads/DBS-main/matlab_data/simulation_results-2.mat)

### `esp32_firmware/`
- Arduino-facing firmware and model headers.
- Most important files:
  - [esp32_firmware/README.md](/Users/maaary/Downloads/DBS-main/esp32_firmware/README.md)
  - [esp32_firmware/dbs_inference.ino](/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_inference.ino)
  - [esp32_firmware/dbs_benchmark/dbs_benchmark.ino](/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_benchmark/dbs_benchmark.ino)
  - [esp32_firmware/dbs_ardutflite_benchmark/dbs_ardutflite_benchmark.ino](/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_ardutflite_benchmark/dbs_ardutflite_benchmark.ino)

### `espidf_firmware/`
- Native ESP-IDF benchmark project for `esp-tflite-micro + ESP-NN`.
- Most important folder:
  - [espidf_firmware/dbs_espnn_benchmark](/Users/maaary/Downloads/DBS-main/espidf_firmware/dbs_espnn_benchmark)

### `models/`
- Trained checkpoints and saved policies.
- Most important folders:
  - [models/TD3_80_80](/Users/maaary/Downloads/DBS-main/models/TD3_80_80)
  - [models/TD3_96_96](/Users/maaary/Downloads/DBS-main/models/TD3_96_96)
  - [models/TD3_128_128](/Users/maaary/Downloads/DBS-main/models/TD3_128_128)
  - [models/policies](/Users/maaary/Downloads/DBS-main/models/policies)

### `results/`
- Active results kept for review and manuscript writing.
- See [results/README.md](/Users/maaary/Downloads/DBS-main/results/README.md).

## Root-Level Files Worth Keeping In Mind

- [states_eval_4d.npy](/Users/maaary/Downloads/DBS-main/states_eval_4d.npy)
  Legacy 4D calibration/eval state file.
- [states_eval_6d.npy](/Users/maaary/Downloads/DBS-main/states_eval_6d.npy)
  Original degenerate 6D file retained as evidence of the earlier calibration bug.
- [states_eval_6d_repaired.npy](/Users/maaary/Downloads/DBS-main/states_eval_6d_repaired.npy)
  Repaired 6D state set used for quantization evaluation.
- [model.h](/Users/maaary/Downloads/DBS-main/model.h)
  Active root-level model header mirrored for firmware flows that still expect it.

## Archived Material

- [archive/repo_cleanup_2026-04-13](/Users/maaary/Downloads/DBS-main/archive/repo_cleanup_2026-04-13)
  Legacy outputs, temporary artifacts, and older notes moved out of the main workflow.
