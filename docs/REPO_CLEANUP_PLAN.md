# Repository Cleanup Plan

## Files to DELETE (Redundant/Old Versions)

### Old/Redundant Python Scripts:
1. `BGN_MC_fixed.py` - Old version, replaced by `BGN_MC.py`
2. `BGN_MC_no_matlab.py` - Old version, replaced by `BGN_MC.py`
3. `quantize_td3.py` - Old quantization script, replaced by `quantize_model.py`
4. `quantize_td3_fixed.py` - Old quantization script
5. `quantize_td3_real.py` - Old quantization script
6. `real_quantization_working.py` - Old quantization script
7. `simple_quantization_test.py` - Test script, redundant
8. `power_measure.py` - macOS version, replaced by `power_profile_windows.py`
9. `collect_calibration_states.py` - Functionality moved to comprehensive_quantization_eval.py
10. `test_working_environment.py` - Redundant with other test scripts
11. `check_tf.py` - Simple check script, not needed

### Redundant Documentation:
12. `SETUP_COMPLETE.md` - Consolidate into README
13. `SETUP_SUCCESS.md` - Consolidate into README
14. `SETUP_INSTRUCTIONS.md` - Consolidate into README
15. `MATLAB_ENGINE_FIX.md` - Consolidate into README
16. `DEPLOYMENT_STATUS.md` - Superseded by DEPLOYMENT_COMPLETE.md
17. `EXAMPLES_NOTEBOOK_ANALYSIS.md` - Analysis doc, can delete
18. `RESULTS_EXPLANATION.md` - Consolidate with SIMPLE_RESULTS_EXPLANATION.md
19. `QUANTIZATION_EVAL_RESULTS.md` - Can consolidate

### Old Results/Images:
20. `quantization_results.png` - Old result image
21. `quantization_results_real.png` - Old result image
22. `quantization_results_final.png` - Old result image

### Temporary Directories:
23. `test_models/` - Empty test directory
24. `__pycache__/` - Python cache (can regenerate)

## Files to KEEP (Essential)

### Core Scripts:
- `BGN_MC.py` - Main environment
- `training.py` - Training script
- `quantize_model.py` - Quantization script
- `comprehensive_quantization_eval.py` - Evaluation script
- `power_profile_windows.py` - Power profiling
- `test_quantized_model.py` - Model testing
- Deployment pipeline: `convert_to_onnx.py`, `convert_onnx_to_tf.py`, `convert_tf_to_tflite.py`, `convert_tflite_to_c.py`

### Test/Verification Scripts:
- `test_matlab_setup.py`
- `test_bgn_environment.py`
- `test_training.py`
- `test_cell1.py`
- `quick_setup_check.py`
- `check_matlab_status.py`
- `verify_examples_workflow.py`
- `setup_environment.py`

### Documentation (Keep Essential):
- `README.md` - Main documentation
- `TODO_LIST.md` - Task tracking
- `SIMPLE_RESULTS_EXPLANATION.md` - Results explanation
- `DEPLOYMENT_FORMATS_EXPLANATION.md` - Format explanations
- `DEPLOYMENT_COMPLETE.md` - Deployment status
- `POWER_PROFILING_RESULTS.md` - Power profiling results
- `MATLAB_USAGE_EXPLANATION.md` - MATLAB usage
- `MSE_EXPLANATION.md` - MSE explanation
- `POWER_ANALYSIS_CORRECTION.md` - Power analysis correction

### Data/Models:
- `models/` - Trained models
- `onnx_actors/`, `tf_model/`, `tflite_actors/` - Deployment outputs
- `model.h` - C header
- `states_eval.npy` - Calibration states
- `bgn_vars.mat` - MATLAB state file
- `logs/` - Training logs

### MATLAB Files:
- `bgn_init.m`, `bgn_step.m` - MATLAB scripts
- `gating/` - MATLAB functions

### Configuration:
- `requirements.txt`
- `examples.ipynb` - Reference notebook
- `install_matlab_engine.bat` - Installation script
- `install_matlab_engine_admin.ps1` - Admin installation script

## Organization Plan

### Create Directories:
- `scripts/` - Move test/verification scripts
- `docs/` - Move documentation files
- `results/` - Move result JSON files and plots

