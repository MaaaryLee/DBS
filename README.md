# Low-Power RL for On-Device Intelligent DBS

This project explores low-power variants of Reinforcement Learning (RL) for on-device intelligent Deep Brain Stimulation (DBS) systems. The system uses a Python-MATLAB framework where the brain model runs in MATLAB and the RL controller runs in Python.

## Project Overview

- **Goal**: Develop and optimize quantized RL models for edge deployment in DBS systems
- **Framework**: Python (RL controller) + MATLAB (BGN brain model)
- **RL Algorithm**: TD3 (Twin Delayed DDPG)
- **Optimization Focus**: Model quantization (FP32 → INT8), power profiling, performance analysis

## Project Structure

```
DBS/
├── Core Scripts:
│   ├── BGN_MC.py                    # Main MATLAB-integrated environment
│   ├── training.py                  # TD3 training script
│   ├── quantize_model.py            # Model quantization (FP32 → INT8)
│   ├── comprehensive_quantization_eval.py  # Comprehensive evaluation
│   └── power_profile_windows.py     # Power profiling tool
│
├── Deployment Pipeline:
│   ├── convert_to_onnx.py           # PyTorch → ONNX
│   ├── convert_onnx_to_tf.py        # ONNX → TensorFlow
│   ├── convert_tf_to_tflite.py      # TensorFlow → TFLite
│   └── convert_tflite_to_c.py       # TFLite → C byte array
│
├── scripts/                         # Test and verification scripts
│   ├── test_matlab_setup.py
│   ├── test_bgn_environment.py
│   ├── test_training.py
│   ├── test_quantized_model.py
│   └── ...
│
├── docs/                            # Documentation
│   ├── SIMPLE_RESULTS_EXPLANATION.md
│   ├── DEPLOYMENT_FORMATS_EXPLANATION.md
│   └── ...
│
├── results/                         # Evaluation results
│   ├── power_profile_*.json
│   ├── quantization_eval_results_*.json
│   └── quantization_eval_plots/
│
├── models/                          # Trained models
│   ├── TD3_32_32/
│   ├── TD3_48_32/
│   ├── TD3_64_32/
│   ├── TD3_64_64/
│   └── policies/
│
├── Deployment Outputs:
│   ├── onnx_actors/                 # ONNX models
│   ├── tf_model/                    # TensorFlow SavedModel
│   ├── tflite_actors/               # TFLite models
│   └── model.h                      # C byte array header
│
├── MATLAB Files:
│   ├── bgn_init.m                   # MATLAB initialization
│   ├── bgn_step.m                   # MATLAB simulation step
│   └── gating/                      # MATLAB gating functions
│
└── Configuration:
    ├── requirements.txt              # Python dependencies
    ├── examples.ipynb               # Reference notebook
    └── install_matlab_engine*.bat/ps1
```

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
   pip install -r requirements.txt
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
python training.py
```
This will train a TD3 agent with configurable hidden layer sizes (default: 32x32).

### 3. Quantize Model
```bash
python quantize_model.py
```

### 4. Evaluate Quantization
```bash
python comprehensive_quantization_eval.py
```

### 5. Profile Power Consumption
```bash
# Windows
python power_profile_windows.py --mode fp32 --duration 30
python power_profile_windows.py --mode int8 --duration 30
```

## Usage Examples

### Basic Environment Usage
```python
from BGN_MC import BGN_MC

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
from BGN_MC import BGN_MC
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
- Dynamic quantization (PyTorch built-in)
- Custom INT8 quantization with proper scaling
- Fidelity measurement (MSE between FP32/INT8)
- Performance evaluation

### 3. Power Profiling
- CPU power measurement (macOS powermetrics)
- Batch inference benchmarking
- FP32 vs INT8 comparison

## Model Architectures

Trained models available:
- **TD3_32_32**: 32 nodes per hidden layer (smallest)
- **TD3_48_32**: 48/32 nodes (medium)
- **TD3_64_32**: 64/32 nodes (medium-large)
- **TD3_64_64**: 64 nodes per layer (largest)

## Deployment Pipeline

The project includes a pipeline for edge deployment:
1. **PyTorch** → Quantized model
2. **ONNX** → Export to ONNX format
3. **TensorFlow** → Convert ONNX to TensorFlow SavedModel
4. **TFLite** → Convert to TensorFlow Lite
5. **C Array** → Generate C byte array for ESP32/microcontrollers

See `examples.ipynb` for detailed conversion steps.

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

## Next Steps

1. ✅ Set up environment (run `setup_environment.py`)
2. ⏭ Train/verify TD3 models work with MATLAB
3. ⏭ Collect calibration states for quantization
4. ⏭ Run comprehensive quantization evaluation
5. ⏭ Profile power consumption (adapt for Windows if needed)
6. ⏭ Test deployment pipeline end-to-end

## References

- BGN Implementation: [IEEE Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10684783&tag=1)
- Stable-Baselines3: [Documentation](https://stable-baselines3.readthedocs.io/)
- PyTorch Quantization: [Documentation](https://pytorch.org/docs/stable/quantization.html)

## License

[Add your license here]

