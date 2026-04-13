# Low-Power RL for On-Device Intelligent DBS

This project explores low-power variants of Reinforcement Learning (RL) for on-device intelligent Deep Brain Stimulation (DBS) systems. The system uses a Python-MATLAB framework where the brain model runs in MATLAB and the RL controller runs in Python.

## Project Overview

- **Goal**: Develop and optimize quantized RL models for edge deployment in DBS systems
- **Framework**: Python (RL controller) + MATLAB (BGN brain model)
- **RL Algorithm**: TD3 (Twin Delayed DDPG)
- **Optimization Focus**: Model quantization (FP32 → INT8), power profiling, performance analysis

## Current Status

- `core/BGN_MC.py` is the full MATLAB Engine environment. It supports:
  - `hvgi` -> 4D observation
  - `hsgi` -> 4D observation
  - `hvgi_sgi` -> 6D observation
- `core/BGN_MC_Online.py` is the cached/offline environment and always returns a 6D observation.
- `core/training.py` currently trains with `BGN_MC_Online`, so newly trained models from that script are 6D.
- Some checked-in legacy checkpoints and deployment artifacts were exported from older 4D flows. In other words, this repo currently contains both 4D and 6D model artifacts.
- Calibration state files should match the model input dimension. Use dimension-specific files such as `states_eval_4d.npy` and `states_eval_6d.npy` instead of assuming one generic `states_eval.npy` works for every model.
- PyTorch INT8 latency is useful as a sanity check, but TFLite and on-device ESP32 measurements are the deployment-relevant benchmarks.

## Project Structure

Key folders:

- `core/`
  Training, environment, and quantization logic.
- `deployment/`
  Export pipeline from trained policy to deployable model formats.
- `scripts/`
  Benchmark, evaluation, and ESP32 helper scripts.
- `results/paper_quantization/`
  Quantization metrics and supporting summaries.
- `results/larger_models/`
  6D size-sweep artifacts and summaries.
- `results/esp32/`
  On-device benchmark traces.
- `esp32_firmware/` and `espidf_firmware/`
  Arduino and native ESP32 deployment paths.
- `archive/repo_cleanup_2026-04-13/`
  Legacy, temporary, and exploratory artifacts kept outside the main workflow.

See [REPO_ORGANIZATION.md](/Users/maaary/Downloads/DBS-main/REPO_ORGANIZATION.md) for a folder map and [results/README.md](/Users/maaary/Downloads/DBS-main/results/README.md) for the main result files.

## Setup Instructions

### Prerequisites

1. **MATLAB** (R2024a or compatible)
   - Must be installed locally
   - MATLAB Engine API for Python must be configured

2. **Python** (3.8+)
   - Virtual environment recommended

### Installation Steps

1. **Install MATLAB Engine for Python**:
   ```bash
   # Navigate to MATLAB's Python engine directory
   cd "C:\Program Files\MATLAB\R2024a\extern\engines\python"
   # Or find your MATLAB installation path
   
   # Install the engine
   python setup.py install
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r config/requirements.txt
   ```

3. **Verify Setup**:
   ```bash
   # Run the comprehensive setup script
   python scripts/setup_environment.py
   
   # Or run individual tests
   python scripts/test_matlab_setup.py      # Test MATLAB connection
   python scripts/test_bgn_environment.py   # Test BGN environment
   ```

## Quick Start

How the system is organized:

- Environment:
  - `core/BGN_MC.py`: full MATLAB-engine driven environment
  - `core/BGN_MC_Online.py`: cached/offline environment backed by `matlab/bgn_vars.mat`
- Training:
- `core/training.py`: TD3 training with configurable hidden sizes
  - Saves checkpoints to `models/TD3_<h1>_<h2>/<timesteps>.zip`
- Quantization and evaluation:
  - `core/quantize_model.py`: exports `actor_fp32_*`, `actor_int8_dynamic_*`, `actor_int8_static_*`
  - `core/comprehensive_quantization_eval.py`: compares fidelity, model size, and latency
- Deployment:
  - `deployment/convert_to_onnx.py`
  - `deployment/convert_onnx_to_tf.py`
  - `deployment/convert_tf_to_tflite.py`
  - `deployment/convert_tflite_to_c.py`
  - Outputs land in `onnx_actors/`, `tf_model/`, `tflite_actors/`, and `esp32_firmware/model_*.h`

### 1. Verify Environment
```bash
python scripts/setup_environment.py
```

### 2. Train a TD3 Model
```bash
python core/training.py
```
This trains a TD3 agent with configurable hidden layer sizes (default: `22x22`) using `BGN_MC_Online`, which currently means a 6D observation space.

### 3. Quantize Model
```bash
python core/quantize_model.py
```

- Exports three actor checkpoints under `models/policies/`:
  - `actor_fp32_*` (baseline)
  - `actor_int8_dynamic_*` (weight-only dynamic quantization)
  - `actor_int8_static_*` (post-training static quantization with calibration states)
- Writes per-variant latency/size metrics to `quantization_summary_*.json`.
- Calibration states must match the actor input dimension. The tooling now prefers dimension-specific files such as `states_eval_4d.npy` and `states_eval_6d.npy`.

### 4. Evaluate Quantization
```bash
python core/comprehensive_quantization_eval.py \
  --variant static_int8 \
  --output-dir results/example_quant_eval/static \
  --skip-env
```

### 5. Measure Latency
```bash
# Measure PyTorch FP32 latency
python scripts/measure_fp32_latency.py

# Measure PyTorch INT8 latency
python scripts/measure_pytorch_int8_latency.py

# Measure TFLite FP32 latency on the deployment-style fast path
python scripts/measure_tflite_fp32_latency.py

# Measure TFLite INT8 latency on the deployment-style fast path
python scripts/measure_tflite_int8_latency.py

# Run the full TFLite comparison (delegates on/off in one report)
python scripts/compare_tflite_variants.py

# Optional: disable delegates to inspect the slower baseline path
python scripts/measure_tflite_fp32_latency.py --disable-default-delegates
python scripts/measure_tflite_int8_latency.py --disable-default-delegates

# Generate latency comparison plots
python scripts/plot_latency_comparison.py
```

Benchmarking notes:

- PyTorch INT8 measurements are a framework-level reference, not the final deployment metric.
- Static PyTorch INT8 can still be slower than FP32 for very small MLPs because quantization/dequantization, packed-parameter access, and kernel dispatch overhead can dominate the saved arithmetic.
- TFLite FP32 vs INT8 is the better software-side comparison for this repo because the deployment target is TFLite Micro / ESP32.
- On Apple Silicon desktops, enabling the default TFLite delegate path is the easiest way to reproduce the faster INT8 result.
- The TFLite latency scripts now default to `inner_repeats=50`, which times a short burst of invokes and divides back down to a per-inference average. This reduces Python timing noise for microsecond-scale models and makes the INT8/FP32 comparison much more stable.
- The FP32 export should be a true float baseline. `deployment/convert_tf_to_tflite.py` now disables `Optimize.DEFAULT` by default so the FP32 file is not silently converted into a hybrid / dynamic-range model.
- The TFLite export scripts now wrap the SavedModel in a static batch=`1` signature before conversion so the generated flatbuffer does not rely on dynamic input shapes.
- Delegate settings matter. Running TFLite with default delegates disabled is useful for apples-to-apples backend comparison, while enabling delegates is closer to a deployment-style performance check and can change which variant is faster.

### 6. Profile Power Consumption
```bash
# Windows
python core/power_profile_windows.py --mode fp32 --duration 30
python core/power_profile_windows.py --mode int8 --duration 30
```

### 7. Deploy to ESP32
```bash
# 1. Prepare the active benchmark model header
python3 scripts/prepare_esp32_benchmark.py --variant int8

# 2. Compile/upload the recommended ESP32 benchmark sketch
python3 scripts/compile_esp32_benchmark.py --upload

# 3. Capture the on-device benchmark result
python3 scripts/run_esp32_benchmark.py --runs 200

# 4. Repeat with the FP32 variant when you want the comparison
python3 scripts/prepare_esp32_benchmark.py --variant fp32
```

Important deployment note:

- The recommended ESP32 benchmark sketch now reads the model input dimension from the embedded flatbuffer and supports both 4D and 6D models up to a max feature size of 8.
- Refresh `esp32_firmware/model.h` with `scripts/prepare_esp32_benchmark.py` before flashing, otherwise you may accidentally benchmark stale artifacts.
- Use matching artifacts end to end: do not flash a 6D model header if the exported TFLite came from a 4D actor, and vice versa.
- The recommended benchmark reports `quantize`, `invoke`, `dequant`, and `total`. `invoke` tells you whether the INT8 kernel is faster; `total` tells you whether deployment is faster once conversion overhead is included.
- If you intentionally want the older ArduTFLite Hjorth-feature pipeline benchmark, use `python3 scripts/compile_esp32_benchmark.py --mode legacy --upload` and `python3 scripts/run_esp32_benchmark.py --mode legacy --timeout 30`.

### 8. Reproduce The Full 96x96-Style Pipeline For A New Model Size
```bash
python3 scripts/build_model_family_artifacts.py \
  --h1 400 \
  --h2 300 \
  --checkpoint-timesteps 500 \
  --train-if-missing
```

This one command is the cleanest way to make a new model family follow the same path we used for `96x96`.

It will:
- train or reuse `models/TD3_<h1>_<h2>/<timesteps>.zip`
- export ONNX into `results/larger_models/<h1>_<h2>/`
- export TensorFlow SavedModel into `results/larger_models/<h1>_<h2>/`
- export `FP32` and `INT8` TFLite models into `results/larger_models/<h1>_<h2>/`
- export ESP32-ready C headers for both TFLite variants
- run desktop TFLite latency benchmarks
- write a manifest summarizing the produced artifacts

Important note:
- A standalone `.tflite` file is already an exported model, not a trainable checkpoint. If you only have a `.tflite` file and no matching training checkpoint / policy weights, the repo can benchmark and deploy it, but it cannot retrain or regenerate a matching INT8 model family from that file alone.

## Usage Examples

### Basic Environment Usage
```python
from core.BGN_MC_Online import BGN_MC_Online

# Create cached/offline environment (current training path, 6D observation)
env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)

# Reset environment
obs, info = env.reset()

# Run simulation
for _ in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

### Training TD3 Agent
```python
from stable_baselines3 import TD3
from core.BGN_MC_Online import BGN_MC_Online
import torch

env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[22, 22], qf=[22, 22])
)

model = TD3('MlpPolicy', env, verbose=1, 
            policy_kwargs=policy_kwargs, 
            learning_rate=0.0001)

model.learn(total_timesteps=2500)
model.save('models/TD3_22_22/2500')
```

### Quantization
```python
from pathlib import Path
from core.quantize_model import quantize_td3_actor

artifacts = quantize_td3_actor(
    hidden_dims=(22, 22),
    model_timesteps=2500,
    states_path=Path("states_eval_6d.npy"),
)

print(artifacts["fp32"].checkpoint)
print(artifacts["dynamic_int8"].checkpoint)
print(artifacts["static_int8"].checkpoint)
```

### Generating Calibration States
```bash
# 6D cached/offline states for BGN_MC_Online models
python scripts/generate_states_eval.py \
  --source online_cached \
  --num-states 1000 \
  --pd \
  --out states_eval_6d.npy

# 4D MATLAB-engine states for legacy BGN_MC models
python scripts/generate_states_eval.py \
  --source matlab_engine \
  --mode hvgi \
  --num-states 1000 \
  --pd \
  --out states_eval_4d.npy
```

## Key Features

### 1. Multiple Observation Modes
- `hvgi`: 4-element state (Hjorth parameters of VGI signal)
- `hsgi`: 4-element state (Hjorth parameters of SGI signal)
- `hvgi_sgi`: 6-element state (combined features)

### 2. Quantization Pipeline
- **Dynamic Quantization**: Weight-only INT8 quantization (PyTorch built-in)
- **Static Quantization**: Post-training static INT8 with calibration states
- **Fidelity Measurement**: MSE between FP32/INT8 outputs
- **Performance Evaluation**: Comprehensive evaluation with action difference analysis
- **Model Size Reduction**: ~21% size reduction with INT8 quantization

### 3. Latency Measurement
- **PyTorch FP32/INT8**: Native PyTorch inference latency
- **TFLite FP32/INT8**: TensorFlow Lite inference latency
- **Comparison Plots**: Visual comparison of latency across formats
- **Batch Processing**: Supports batch inference for throughput analysis

### 4. Power Profiling
- **CPU Power Measurement**: Platform-specific power profiling (Windows/macOS)
- **Batch Inference Benchmarking**: Power consumption during batch processing
- **FP32 vs INT8 Comparison**: Power efficiency analysis
- **Memory Profiling**: Heap usage tracking

### 5. Edge Deployment
- **ESP32 Support**: Full firmware for ESP32 microcontrollers
- **TFLite Integration**: Optimized TensorFlow Lite models
- **C Array Export**: Direct model embedding for microcontrollers
- **Real-time Inference**: Low-latency on-device inference
- **Power Monitoring**: Built-in power and memory profiling on ESP32

## Model Architectures

Trained models available:
- **TD3_22_22**: 22 nodes per hidden layer (**preferred / smallest**)
- **TD3_32_32**: 32 nodes per hidden layer
- **TD3_48_32**: 48/32 nodes (medium)
- **TD3_64_32**: 64/32 nodes (medium-large)
- **TD3_64_64**: 64 nodes per layer (largest)

## Deployment Pipeline

The project includes a complete pipeline for edge deployment:

1. **Training**: Train TD3 agent with PyTorch/Stable-Baselines3
2. **Quantization**: Convert FP32 → INT8 (dynamic or static)
3. **ONNX Export**: PyTorch → ONNX format
4. **TensorFlow Conversion**: ONNX → TensorFlow SavedModel
5. **TFLite Conversion**: TensorFlow → TensorFlow Lite (FP32 or INT8)
6. **C Array Generation**: TFLite → C byte array header
7. **ESP32 Deployment**: Upload firmware with embedded model

### Model Formats Available

- **PyTorch**: `.pt` files (FP32, INT8 dynamic, INT8 static)
- **ONNX**: `.onnx` files (intermediate format)
- **TensorFlow**: SavedModel format
- **TFLite**: `.tflite` files (FP32 and INT8 variants)
- **C Array**: active `esp32_firmware/model.h` for the current benchmark target, plus optional legacy `esp32_firmware/model_fp32.h` / `esp32_firmware/model_int8.h`

### Deployment Workflow

```bash
# 1. Train model
python core/training.py

# 2. Quantize model
python core/quantize_model.py

# 3. Convert to ONNX
python deployment/convert_to_onnx.py

# 4. Convert to TensorFlow
python deployment/convert_onnx_to_tf.py

# 5. Convert to TFLite
python deployment/convert_tf_to_tflite.py

# 6. Generate C array
python deployment/convert_tflite_to_c.py

# 7. Deploy to ESP32
# See esp32_firmware/README.md
```

See `notebooks/examples.ipynb` for detailed conversion steps and `docs/DEPLOYMENT_COMPLETE.md` for deployment status.

## Troubleshooting

### Waiting Updates
### MATLAB Engine Issues
- **Error**: "Cannot find MATLAB"
  - Solution: Add MATLAB to system PATH or specify MATLAB installation path
  - Verify: `python test_matlab_setup.py`

- **Error**: "Module 'matlab' not found"
  - Solution: Install MATLAB Engine: `cd <MATLAB>/extern/engines/python && python setup.py install`
  - Or: `pip install matlabengine==24.2.1`

### Environment Issues
- **Error**: "bgn_init.m not found"
  - Solution: Ensure MATLAB working directory is set correctly (now auto-detected)
  - Verify all `.m` files are in the workspace directory

### Quantization Issues
- **Error**: "States file not found"
  - Solution: The evaluation script will automatically collect calibration states if needed

## Project Status

### Completed
- [x] Environment setup and MATLAB integration
- [x] TD3 training with multiple architectures
- [x] Model quantization (FP32 → INT8)
- [x] Comprehensive quantization evaluation
- [x] Latency measurement across formats
- [x] Power profiling tools
- [x] Complete deployment pipeline (PyTorch → ESP32)
- [x] ESP32 firmware with inference and profiling

### In Progress / Future Work
- [ ] Real-time sensor integration on ESP32
- [ ] Hardware power measurement (INA219/INA260)
- [ ] Online learning capabilities
- [ ] Multi-device deployment testing
- [ ] Clinical validation

## Results Summary

- **Model Size**: 5.7 KB (TFLite INT8) vs 8.9 KB (TFLite FP32) - 36% reduction
- **Quantization Accuracy**: <1% MSE difference between FP32 and INT8
- **Latency**: See `results/latency_comparison_*.png` for detailed comparisons
- **Power Efficiency**: INT8 shows ~20-30% power reduction in profiling

See `docs/SIMPLE_RESULTS_EXPLANATION.md` for detailed results analysis.
