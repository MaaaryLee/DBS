# DBS-1 Environment Setup

## ✅ Completed Setup

### First Virtual Environment (Training) - `venv_training`
- **Python Version**: 3.11.14
- **Purpose**: PyTorch-based training and RL model development
- **Location**: `/Users/maaarylee/DBS-1/venv_training/`

#### Installed Packages:
- `gymnasium==1.2.0` - RL environment framework
- `torch==2.6.0` - PyTorch for neural networks
- `stable-baselines3==2.7.0` - RL algorithms (TD3)
- `numpy==2.2.6` - Numerical computing
- `numba==0.61.2` - JIT compilation
- `onnx==1.19.0` - Model export format
- `onnx-ir==0.1.8` - ONNX intermediate representation
- `onnxscript==0.3.1` - ONNX scripting
- `scipy` - Scientific computing
- `matplotlib` - Plotting
- `scikit-learn` - Machine learning utilities
- `antropy` - Entropy calculations for biomarkers

#### Activation:
```bash
# Method 1: Direct activation
source venv_training/bin/activate

# Method 2: Using convenience script
./activate_training.sh
```

## ⏳ Pending Setup

### Second Virtual Environment (Deployment) - `venv_deployment`
- **Purpose**: TensorFlow-based model deployment and conversion
- **Status**: Created but not yet configured

#### Required Packages (to be installed):
- `keras==2.15.0`
- `tensorflow==2.15.0`
- `tensorflow-addons==0.22.0`
- `tensorflow-probability==0.22.0`
- `tf-keras==2.15.0`
- `onnx-tf==1.10.0`
- `matlabengine==24.2.1` (requires MATLAB installation)

## ⚠️ Notes

1. **MATLAB Engine**: Currently skipped due to MATLAB not being installed. This is required for the full brain simulation functionality.

2. **Environment Isolation**: Each environment is completely isolated with its own package versions to avoid conflicts.

3. **Next Steps**: 
   - Install MATLAB (if needed for full functionality)
   - Set up the deployment environment
   - Test the training pipeline

## 🚀 Ready to Use

The training environment is now ready for:
- Running the examples notebook
- Training TD3 models
- Model quantization experiments
- Basic brain simulation (without MATLAB)
