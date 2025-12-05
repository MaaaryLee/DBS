# Low-Power RL for On-Device Intelligent DBS

This project explores low-power variants of Reinforcement Learning (RL) for on-device intelligent Deep Brain Stimulation (DBS) systems. The system uses a Python-MATLAB framework where the brain model runs in MATLAB and the RL controller runs in Python.

## Project Overview

- **Goal**: Develop and optimize quantized RL models for edge deployment in DBS systems
- **Framework**: Python (RL controller) + MATLAB (BGN brain model)
- **RL Algorithm**: TD3 (Twin Delayed DDPG)
- **Optimization Focus**: Model quantization (FP32 → INT8), power profiling, performance analysis

## Project Structure

```
DBS-1/
├── core/                           # Core training and evaluation scripts
│   ├── BGN_MC.py                   # Main MATLAB-integrated environment
│   ├── BGN_MC_Online.py            # Online MATLAB workflow
│   ├── training.py                  # TD3 training script
│   ├── quantize_model.py           # Model quantization (FP32 → INT8)
│   ├── comprehensive_quantization_eval.py  # Comprehensive evaluation
│   ├── power_profile_windows.py    # Power profiling tool
│   └── ...
│
├── deployment/                     # Model deployment conversion scripts
│   ├── convert_to_onnx.py          # PyTorch → ONNX
│   ├── convert_onnx_to_tf.py       # ONNX → TensorFlow
│   ├── convert_tf_to_tflite.py     # TensorFlow → TFLite
│   └── convert_tflite_to_c.py      # TFLite → C byte array
│
├── matlab/                         # MATLAB simulation files
│   ├── bgn_init.m                  # MATLAB initialization
│   ├── bgn_step.m                  # MATLAB simulation step
│   ├── bgn_vars.mat                # MATLAB state variables
│   └── gating/                      # MATLAB gating functions
│
├── matlab_data/                    # MATLAB simulation data
│   ├── run_simulation_online.m
│   └── simulation_results.mat
│
├── config/                         # Configuration files
│   ├── requirements.txt            # Python dependencies
│   ├── requirements_training.txt  # Training dependencies
│   ├── requirements_deployment.txt # Deployment dependencies
│   └── install_matlab_engine*.bat/ps1
│
├── notebooks/                      # Jupyter notebooks
│   ├── examples.ipynb              # Main examples
│   └── bgnm_testing.ipynb
│
├── scripts/                        # Utility and test scripts
│   ├── setup_environment.py        # Environment setup
│   ├── test_*.py                   # Test scripts
│   ├── measure_*_latency.py        # Latency measurement
│   └── ...
│
├── esp32_firmware/                 # ESP32 deployment
│   ├── dbs_inference.ino           # Main inference firmware
│   ├── model.h                      # TFLite model as C array
│   └── README.md
│
├── docs/                           # Documentation
│   └── ...
│
├── results/                        # Evaluation results
│   ├── *_latency.json              # Latency measurements
│   ├── latency_comparison_*.png     # Comparison plots
│   └── quant_eval_run*/            # Evaluation runs
│
├── models/                         # Trained models
│   ├── TD3_32_32/                  # Model checkpoints
│   └── policies/                   # Actor checkpoints
│
├── onnx_actors/                    # ONNX models
├── tf_model/                       # TensorFlow SavedModel
└── tflite_actors/                  # TFLite models
```

**Note**: The repository has been reorganized for better clarity. See `REPO_ORGANIZATION.md` for detailed structure.

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

### 1. Verify Environment
```bash
python scripts/setup_environment.py
```

### 2. Train a TD3 Model
```bash
python core/training.py
```
This will train a TD3 agent with configurable hidden layer sizes (default: 32x32).

### 3. Quantize Model
```bash
python core/quantize_model.py
```

- Exports three actor checkpoints under `models/policies/`:
  - `actor_fp32_*` (baseline)
  - `actor_int8_dynamic_*` (weight-only dynamic quantization)
  - `actor_int8_static_*` (post-training static quantization with calibration states)
- Writes per-variant latency/size metrics to `quantization_summary_*.json`.

### 4. Evaluate Quantization
```bash
python core/comprehensive_quantization_eval.py \
  --variant static_int8 \
  --output-dir results/quant_eval_run2/static \
  --skip-env
```

### 5. Measure Latency
```bash
# Measure PyTorch FP32 latency
python scripts/measure_fp32_latency.py

# Measure PyTorch INT8 latency
python scripts/measure_pytorch_int8_latency.py

# Measure TFLite FP32 latency
python scripts/measure_tflite_fp32_latency.py

# Measure TFLite INT8 latency
python scripts/measure_tflite_int8_latency.py

# Generate latency comparison plots
python scripts/plot_latency_comparison.py
```

### 6. Profile Power Consumption
```bash
# Windows
python core/power_profile_windows.py --mode fp32 --duration 30
python core/power_profile_windows.py --mode int8 --duration 30
```

### 7. Deploy to ESP32
```bash
# 1. Convert model to C array (if not already done)
python deployment/convert_tflite_to_c.py

# 2. Copy model to ESP32 firmware directory
cp model.h esp32_firmware/

# 3. Open esp32_firmware/dbs_inference.ino in Arduino IDE
# 4. Upload to ESP32 board
# See esp32_firmware/README.md for detailed instructions
```

## Usage Examples

### Basic Environment Usage
```python
from core.BGN_MC import BGN_MC

# Create environment (Parkinsonian state)
env = BGN_MC(tmax=1100, pd=True, mode='hvgi_sgi')

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
from core.BGN_MC import BGN_MC
import torch

env = BGN_MC(tmax=1100, pd=True)
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[32, 32], qf=[32, 32])
)

model = TD3('MlpPolicy', env, verbose=1, 
            policy_kwargs=policy_kwargs, 
            learning_rate=0.0001)

model.learn(total_timesteps=2500)
model.save('models/TD3_32_32/2500')
```

### Quantization
```python
import torch
from torch.ao.quantization import quantize_dynamic
from stable_baselines3 import TD3

# Load trained model
model = TD3.load('models/TD3_32_32/2500.zip')
policy = model.policy.to(torch.device('cpu'))
policy.eval()

# Quantize to INT8
qpolicy = quantize_dynamic(policy, dtype=torch.qint8)

# Save quantized model
torch.save(qpolicy.state_dict(), 'models/policies/qpolicy_32_32.pth')
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
- **TD3_32_32**: 32 nodes per hidden layer (smallest)
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
- **C Array**: `model.h` header file for microcontrollers

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

### ✅ Completed
- [x] Environment setup and MATLAB integration
- [x] TD3 training with multiple architectures
- [x] Model quantization (FP32 → INT8)
- [x] Comprehensive quantization evaluation
- [x] Latency measurement across formats
- [x] Power profiling tools
- [x] Complete deployment pipeline (PyTorch → ESP32)
- [x] ESP32 firmware with inference and profiling

### 🔄 In Progress / Future Work
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

## References

- BGN Implementation: [IEEE Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10684783&tag=1)
- Stable-Baselines3: [Documentation](https://stable-baselines3.readthedocs.io/)
- PyTorch Quantization: [Documentation](https://pytorch.org/docs/stable/quantization.html)

## License

[Add your license here]

