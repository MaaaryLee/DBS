# Repository Organization

## Directory Structure

```
DBS/
├── BGN_MC.py                    # Main MATLAB-integrated environment
├── training.py                  # TD3 training script
├── quantize_model.py            # Model quantization (FP32 → INT8)
├── comprehensive_quantization_eval.py  # Comprehensive evaluation
├── power_profile_windows.py     # Power profiling tool
├── test_quantized_model.py      # Test quantized model performance
│
├── Deployment Pipeline:
│   ├── convert_to_onnx.py       # PyTorch → ONNX
│   ├── convert_onnx_to_tf.py    # ONNX → TensorFlow
│   ├── convert_tf_to_tflite.py  # TensorFlow → TFLite
│   └── convert_tflite_to_c.py   # TFLite → C byte array
│
├── scripts/                     # Test and verification scripts
│   ├── test_matlab_setup.py
│   ├── test_bgn_environment.py
│   ├── test_training.py
│   ├── test_cell1.py
│   ├── quick_setup_check.py
│   ├── check_matlab_status.py
│   ├── setup_environment.py
│   └── verify_examples_workflow.py
│
├── docs/                        # Documentation
│   ├── SIMPLE_RESULTS_EXPLANATION.md
│   ├── DEPLOYMENT_FORMATS_EXPLANATION.md
│   ├── DEPLOYMENT_COMPLETE.md
│   ├── POWER_PROFILING_RESULTS.md
│   ├── MATLAB_USAGE_EXPLANATION.md
│   ├── MSE_EXPLANATION.md
│   └── POWER_ANALYSIS_CORRECTION.md
│
├── results/                     # Evaluation results
│   ├── power_profile_fp32_32_32.json
│   ├── power_profile_int8_32_32.json
│   ├── quantization_eval_results_32_32.json
│   └── quantization_eval_plots/
│
├── models/                      # Trained models
│   ├── TD3_32_32/
│   ├── TD3_48_32/
│   ├── TD3_64_32/
│   ├── TD3_64_64/
│   └── policies/
│
├── Deployment Outputs:
│   ├── onnx_actors/            # ONNX models
│   ├── tf_model/               # TensorFlow SavedModel
│   ├── tflite_actors/          # TFLite models
│   └── model.h                  # C byte array header
│
├── MATLAB Files:
│   ├── bgn_init.m              # MATLAB initialization
│   ├── bgn_step.m               # MATLAB simulation step
│   ├── bgn_vars.mat            # MATLAB state file
│   └── gating/                  # MATLAB gating functions
│
├── Configuration:
│   ├── requirements.txt         # Python dependencies
│   ├── examples.ipynb          # Reference notebook
│   ├── install_matlab_engine.bat
│   └── install_matlab_engine_admin.ps1
│
├── Data:
│   ├── states_eval.npy          # Calibration states
│   └── logs/                    # Training logs
│
└── README.md                    # Main documentation
```

## Key Files

### Core Scripts
- **BGN_MC.py**: Main environment integrating MATLAB brain model with Python RL
- **training.py**: Train TD3 models with configurable architectures
- **quantize_model.py**: Convert FP32 models to INT8
- **comprehensive_quantization_eval.py**: Full evaluation pipeline
- **power_profile_windows.py**: CPU usage and inference time profiling

### Deployment Pipeline
All conversion scripts follow the pipeline: PyTorch → ONNX → TensorFlow → TFLite → C

### Test Scripts (in `scripts/`)
All verification and testing scripts are organized in the `scripts/` directory

### Documentation (in `docs/`)
All explanatory and result documentation is in the `docs/` directory

### Results (in `results/`)
All evaluation results, JSON files, and plots are in the `results/` directory

