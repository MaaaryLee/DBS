# Low-Power RL for On-Device Intelligent DBS

This repository studies reinforcement-learning controllers for intelligent deep brain stimulation (DBS), with an emphasis on quantization and embedded deployment.

The project combines:

- a MATLAB basal ganglia simulator
- a Python TD3 controller
- export and quantization tooling
- desktop TFLite benchmarks
- ESP32 deployment benchmarks

## Current Focus

The active research line in this repo is the **6D deployment path**.

- `core/BGN_MC_Online.py` returns a 6D observation and is the current training path.
- The native deployment benchmark is:
  `ESP-IDF + esp-tflite-micro + ESP-NN`
- The main on-device candidate at the moment is **6D `96x96` INT8**.

Current headline result:

- On native ESP32-S3 with the default `32 KB` data-cache configuration, the 6D `96x96` INT8 model was faster than FP32 on-device while remaining close to FP32 on held-out fidelity and replay-based control agreement tests.

The repo still contains some older **4D** artifacts. Those are kept for reference, but the main quantization/deployment story is now centered on **6D** models.

## Start Here

If you only need the main project entry points:

- [REPO_ORGANIZATION.md](/Users/maaary/Downloads/DBS-main/REPO_ORGANIZATION.md)
  concise map of the active folders
- [results/README.md](/Users/maaary/Downloads/DBS-main/results/README.md)
  guide to the result files worth citing
- [quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)
  main quantitative summary
- [pi_findings_since_96x96_threshold_2026-04-15.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/pi_findings_since_96x96_threshold_2026-04-15.md)
  PI-facing research summary

## Repository Layout

Main folders:

- `core/`
  training, environments, quantization logic
- `deployment/`
  export pipeline from checkpoint to ONNX / TensorFlow / TFLite / C header
- `scripts/`
  repeatable benchmark, evaluation, and ESP32 helper scripts
- `esp32_firmware/`
  Arduino-facing benchmark sketch and active `model.h`
- `espidf_firmware/`
  native ESP-IDF benchmark project for the paper-faithful runtime path
- `models/`
  training checkpoints and exported policy artifacts
- `results/`
  active benchmark and evaluation outputs
- `archive/`
  older notes, legacy outputs, and exploratory material kept out of the main workflow

## Environment and Model Notes

There are two important observation families in this repo:

- **4D**
  legacy flows from older `BGN_MC.py` configurations
- **6D**
  current cached/offline training and deployment flow

Use matching artifacts end to end:

- 4D model -> 4D calibration/eval states -> 4D deployment header
- 6D model -> 6D calibration/eval states -> 6D deployment header

Do not mix them in one benchmark table.

## Minimal Setup

Requirements:

- Python 3.8+
- MATLAB with MATLAB Engine for Python
- ESP-IDF only if you want the native ESP32 benchmark path
- Arduino IDE / `arduino-cli` only if you want the Arduino benchmark path

Install Python dependencies:

```bash
pip install -r config/requirements.txt
```

Validate the MATLAB/Python environment:

```bash
python scripts/setup_environment.py
```

## Main Workflows

### 1. Train a model

```bash
python core/training.py
```

This uses the cached/offline environment and currently produces **6D** models.

### 2. Build a full model family

This is the cleanest way to reproduce the current pipeline for a chosen hidden size:

```bash
python3 scripts/build_model_family_artifacts.py \
  --h1 96 \
  --h2 96 \
  --checkpoint-timesteps 500 \
  --train-if-missing
```

This pipeline can:

- train or reuse a checkpoint
- export ONNX
- export TensorFlow SavedModel
- export FP32 and INT8 TFLite
- generate ESP32-ready C headers
- run desktop TFLite latency benchmarks
- write a per-family manifest

### 3. Benchmark desktop TFLite

```bash
python scripts/measure_tflite_fp32_latency.py
python scripts/measure_tflite_int8_latency.py
python scripts/compare_tflite_variants.py
```

Interpretation:

- desktop TFLite is the **software-side** benchmark
- it is useful, but it is **not** the final deployment metric

### 4. Benchmark native ESP32 deployment

This is the main deployment-relevant path:

- runtime: `ESP-IDF + esp-tflite-micro + ESP-NN`
- benchmark project:
  [dbs_espnn_benchmark](/Users/maaary/Downloads/DBS-main/espidf_firmware/dbs_espnn_benchmark)

Use:

```bash
python3 scripts/run_espidf_benchmark_variant.py \
  --model-path results/larger_models/96_96/model_int8_96_96_repaired.tflite \
  --label native_int8_96_96 \
  --port /dev/cu.usbmodem... \
  --repeats 5
```

### 5. Use the Arduino sketch when needed

Arduino is still useful for manual checks and power measurements.

Recommended sketch:

- [dbs_benchmark.ino](/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_benchmark/dbs_benchmark.ino)

Important:

- the sketch is generic
- the active model is whatever is currently stored in:
  [esp32_firmware/model.h](/Users/maaary/Downloads/DBS-main/esp32_firmware/model.h)

To switch the active Arduino model:

```bash
python3 scripts/prepare_esp32_benchmark.py --variant int8
python3 scripts/prepare_esp32_benchmark.py --variant fp32
```

The native ESP-IDF benchmark remains the preferred path for manuscript-quality latency numbers.

## What to Report

For this project, report results in this order:

1. **native ESP32 benchmark**
   deployment-relevant result
2. **desktop TFLite benchmark**
   supporting software-side comparison
3. **PyTorch INT8/FP32**
   framework reference only, not the main deployment claim

## Current Result Pointers

Most important current result files:

- [quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)
- [summary_6d_models.json](/Users/maaary/Downloads/DBS-main/results/larger_models/summary_6d_models.json)
- [native_fp32_96_96_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_summary.json)
- [native_int8_96_96_repaired_32k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json)
- [eval_tflite_96_96_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/eval_tflite_96_96_strict_holdout.json)
- [pi_findings_since_96x96_threshold_2026-04-15.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/pi_findings_since_96x96_threshold_2026-04-15.md)

## Caveats

- `96x96` is the strongest current deployment candidate, not a universal optimum.
- Cache configuration matters for interpretation.
- The large `400x300` model is useful as a scaling study, but its speedup is smaller and appears more memory-limited.
- Held-out fidelity and replay-based agreement are strong supporting evidence, but they are not the same as a fresh closed-loop stochastic control study.

## Archived Material

Legacy and exploratory artifacts were moved to:

- [archive/repo_cleanup_2026-04-13](/Users/maaary/Downloads/DBS-main/archive/repo_cleanup_2026-04-13)

They are kept for traceability, but they are not the main workflow.
