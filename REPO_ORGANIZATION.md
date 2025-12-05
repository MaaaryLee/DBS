# Repository Organization

## Directory Structure

```
DBS-1/
├── core/                           # Core training and evaluation scripts
│   ├── BGN_MC.py                   # Main MATLAB-integrated environment
│   ├── BGN_MC_Online.py            # Online MATLAB workflow variant
│   ├── training.py                 # TD3 training script
│   ├── quantize_model.py           # Model quantization (FP32 → INT8)
│   ├── comprehensive_quantization_eval.py  # Comprehensive evaluation
│   ├── power_profile_windows.py    # Power profiling tool
│   ├── matlab_online_workflow.py  # MATLAB online integration
│   ├── controller_validation.py   # Controller validation
│   ├── example_matlab_online.py   # MATLAB online example
│   └── test_fresh_dynamics.py      # Dynamics testing
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
│       ├── activations.m
│       ├── gpe_*.m                 # GPE gating functions
│       ├── stn_*.m                 # STN gating functions
│       └── th_*.m                  # Thalamus gating functions
│
├── matlab_data/                    # MATLAB simulation data and results
│   ├── run_simulation_online.m     # Online simulation script
│   ├── simulation_params.json      # Simulation parameters
│   ├── simulation_results.mat      # Simulation results
│   └── MATLAB_Online_Instructions.md
│
├── config/                         # Configuration and setup files
│   ├── requirements.txt            # Main Python dependencies
│   ├── requirements_training.txt   # Training-specific dependencies
│   ├── requirements_deployment.txt # Deployment-specific dependencies
│   ├── install_matlab_engine.bat   # Windows MATLAB engine installer
│   ├── install_matlab_engine_admin.ps1  # Admin installer script
│   └── activate_training.sh        # Training environment activation
│
├── notebooks/                      # Jupyter notebooks
│   ├── examples.ipynb              # Main examples and tutorials
│   └── bgnm_testing.ipynb          # BGN model testing notebook
│
├── assets/                         # Images, plots, and temporary files
│   ├── PI_Presentation_Results.png
│   ├── quantization_results.png
│   ├── system_validation_results.png
│   ├── power_profile_fp32_32_32.json
│   ├── power_profile_int8_32_32.json
│   ├── states_eval.npy             # Calibration states
│   └── gating.zip                   # Gating functions archive
│
├── scripts/                        # Utility and test scripts
│   ├── setup_environment.py        # Environment setup verification
│   ├── test_matlab_setup.py        # MATLAB connection test
│   ├── test_bgn_environment.py     # BGN environment test
│   ├── test_training.py            # Training verification
│   ├── test_quantized_model.py     # Quantization test
│   ├── measure_fp32_latency.py     # FP32 latency measurement
│   ├── measure_pytorch_int8_latency.py  # PyTorch INT8 latency
│   ├── measure_tflite_fp32_latency.py   # TFLite FP32 latency
│   ├── measure_tflite_int8_latency.py   # TFLite INT8 latency
│   ├── plot_latency_comparison.py  # Latency comparison plots
│   ├── convert_saved_model_to_tflite_int8.py  # INT8 TFLite conversion
│   ├── verify_examples_workflow.py # End-to-end workflow verification
│   ├── quick_setup_check.py        # Quick environment check
│   ├── check_matlab_status.py      # MATLAB status checker
│   └── test_cell1.py               # Cell testing script
│
├── esp32_firmware/                 # ESP32 deployment firmware
│   ├── dbs_inference.ino           # Main inference firmware
│   ├── power_monitor.ino           # Power monitoring firmware
│   ├── model.h                     # TFLite model as C array
│   └── README.md                   # ESP32 setup instructions
│
├── docs/                           # Documentation
│   ├── SIMPLE_RESULTS_EXPLANATION.md
│   ├── DEPLOYMENT_FORMATS_EXPLANATION.md
│   ├── DEPLOYMENT_COMPLETE.md
│   ├── ENVIRONMENT_SETUP.md
│   ├── POWER_PROFILING_RESULTS.md
│   ├── MATLAB_USAGE_EXPLANATION.md
│   ├── MSE_EXPLANATION.md
│   ├── POWER_ANALYSIS_CORRECTION.md
│   ├── REPO_CLEANUP_PLAN.md
│   └── TODO_LIST.md
│
├── results/                        # Evaluation results and outputs
│   ├── power_profile_*.json        # Power profiling data
│   ├── *_latency.json              # Latency measurements
│   ├── latency_comparison_*.png    # Latency comparison plots
│   ├── quantization_eval_results_*.json  # Quantization evaluation
│   ├── quant_eval_run*/            # Quantization evaluation runs
│   │   ├── plots/                  # Evaluation plots
│   │   └── *.json                   # Evaluation results
│   └── quantization_eval_plots/    # Legacy plots
│
├── models/                         # Trained models and checkpoints
│   ├── TD3_32_32/                  # 32x32 hidden layers
│   ├── TD3_48_32/                  # 48x32 hidden layers
│   ├── TD3_64_32/                  # 64x32 hidden layers
│   ├── TD3_64_64/                  # 64x64 hidden layers
│   └── policies/                    # Actor checkpoints
│       ├── actor_fp32_*.pt         # FP32 actors
│       ├── actor_int8_dynamic_*.pt # Dynamic INT8 actors
│       ├── actor_int8_static_*.pt  # Static INT8 actors
│       └── quantization_summary_*.json  # Quantization summaries
│
├── onnx_actors/                    # ONNX model exports
│   └── model.onnx
│
├── tf_model/                       # TensorFlow SavedModel
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
│
├── tflite_actors/                 # TensorFlow Lite models
│   ├── model.tflite                # Original TFLite
│   ├── model_fp32.tflite           # FP32 TFLite
│   └── model_int8.tflite           # INT8 TFLite
│
├── logs/                          # Training logs (TensorBoard)
│   └── TD3_32_32_0/
│       └── events.out.tfevents.*
│
├── temp_eval/                      # Temporary evaluation results
│   ├── plots/
│   └── quantization_eval_results_*.json
│
├── README.md                       # Main project documentation
└── REPO_ORGANIZATION.md            # This file
```

## Directory Descriptions

### `core/` - Core Scripts
Contains the main training, evaluation, and environment scripts:
- **BGN_MC.py**: Main environment integrating MATLAB brain model with Python RL
- **training.py**: Train TD3 models with configurable architectures
- **quantize_model.py**: Convert FP32 models to INT8 (dynamic and static)
- **comprehensive_quantization_eval.py**: Full evaluation pipeline
- **power_profile_windows.py**: CPU usage and inference time profiling

### `deployment/` - Deployment Pipeline
Model format conversion scripts following the pipeline: PyTorch → ONNX → TensorFlow → TFLite → C
- Each script handles a specific conversion step
- Final output is `model.h` C array for microcontroller deployment

### `matlab/` - MATLAB Simulation Files
MATLAB scripts and functions for the BGN brain model:
- Core simulation functions (`bgn_init.m`, `bgn_step.m`)
- Gating functions for different brain regions
- State variables and initialization data

### `matlab_data/` - MATLAB Data
Simulation data, parameters, and results from MATLAB runs

### `config/` - Configuration Files
Setup and dependency files:
- Python requirements for different use cases
- MATLAB Engine installation scripts
- Environment activation scripts

### `notebooks/` - Jupyter Notebooks
Interactive notebooks for examples, tutorials, and testing

### `assets/` - Assets and Temporary Files
Images, plots, calibration data, and other temporary files

### `scripts/` - Utility Scripts
Test, verification, and measurement scripts:
- Environment setup and testing
- Latency measurement tools
- Workflow verification

### `esp32_firmware/` - ESP32 Deployment
Complete firmware for ESP32 microcontroller deployment

### `docs/` - Documentation
All project documentation, guides, and explanations

### `results/` - Evaluation Results
All evaluation outputs, measurements, and plots organized by evaluation run

### `models/` - Trained Models
Trained TD3 models and actor checkpoints organized by architecture

### Deployment Outputs
- **onnx_actors/**: ONNX model exports
- **tf_model/**: TensorFlow SavedModel format
- **tflite_actors/**: TensorFlow Lite models (FP32 and INT8)

### `logs/` - Training Logs
TensorBoard event files for training visualization

## File Organization Principles

1. **Separation of Concerns**: Core scripts, deployment, and utilities are clearly separated
2. **Logical Grouping**: Related files are grouped together (e.g., all MATLAB files in `matlab/`)
3. **Clear Naming**: Directory names clearly indicate their purpose
4. **Minimal Root Directory**: Only essential files (README.md, REPO_ORGANIZATION.md) in root
5. **Consistent Structure**: Similar types of files follow the same organizational pattern

## Usage Guidelines

- **Training**: Use scripts from `core/` directory
- **Deployment**: Use scripts from `deployment/` directory in sequence
- **Testing**: Use scripts from `scripts/` directory
- **Configuration**: Check `config/` for setup files
- **Documentation**: Refer to `docs/` for detailed guides
