# TODO List: Low-Power RL for On-Device Intelligent DBS

## ✅ Completed Tasks

1. **Fix hardcoded MATLAB path in BGN_MC.py** ✓
   - Updated to use dynamic workspace directory detection
   - No more hardcoded paths - works on any system

2. **Create MATLAB engine connection test script** ✓
   - `test_matlab_setup.py` - Verifies MATLAB installation and connection
   - Tests MATLAB engine, workspace directory, required files, and gating functions

3. **Update requirements.txt** ✓
   - Added matplotlib and scikit-learn for visualization and metrics
   - All dependencies documented

4. **Create environment setup verification script** ✓
   - `setup_environment.py` - Comprehensive setup verification
   - Tests Python packages, MATLAB connection, and BGN environment

5. **Create README.md** ✓
   - Complete documentation with setup instructions, usage examples, and troubleshooting

## 🔄 Pending Tasks

### Phase 1: Environment Verification (Next Steps)

6. **Verify all MATLAB .m files are accessible and gating functions work**
   - Run `python test_matlab_setup.py` to verify
   - Ensure all gating functions in `gating/` directory are accessible
   - Test that MATLAB can find and execute `bgn_init.m` and `bgn_step.m`

7. **Test training pipeline with MATLAB engine**
   - Verify TD3 training works with MATLAB-integrated environment
   - Test with small number of timesteps first (e.g., 100)
   - Ensure models save correctly
   - Check TensorBoard logging works

### Phase 2: Quantization Framework

8. **Create calibration state collection script**
   - Check if `states_eval.npy` exists
   - If missing, run `python collect_calibration_states.py`
   - Verify states are collected from MATLAB-integrated environment
   - Ensure states are representative of actual environment

9. **Set up quantization testing framework**
   - Test FP32 vs INT8 comparison
   - Verify quantization works with MATLAB environment
   - Compare accuracy metrics (MSE, action differences)
   - Test with multiple model architectures (32x32, 48x32, 64x32, 64x64)

10. **Create comprehensive quantization evaluation script**
    - Accuracy comparison (FP32 vs INT8)
    - Model size reduction analysis
    - Performance benchmarking (inference time)
    - Power consumption comparison (if possible on Windows)
    - Generate comparison plots and reports

### Phase 3: Power Profiling

11. **Set up power profiling tools**
    - Adapt `power_measure.py` for Windows (currently macOS-only)
    - Windows alternatives: Windows Performance Toolkit, Intel Power Gadget, or CPU-Z
    - Create Windows-compatible power measurement script
    - Compare FP32 vs INT8 power consumption
    - Measure inference time differences

### Phase 4: Deployment Pipeline

12. **Test deployment pipeline (PyTorch → ONNX → TFLite)**
    - Export quantized model to ONNX format
    - Convert ONNX to TensorFlow SavedModel
    - Convert SavedModel to TFLite
    - Generate C byte array for ESP32/microcontroller deployment
    - Verify model works in edge deployment format

## 📋 Quick Start Checklist

Use this checklist to verify your setup:

- [ ] Install MATLAB Engine for Python
  ```bash
  cd "<MATLAB_INSTALL_DIR>\extern\engines\python"
  python setup.py install
  ```

- [ ] Install Python dependencies
  ```bash
  pip install -r requirements.txt
  ```

- [ ] Run setup verification
  ```bash
  python setup_environment.py
  ```

- [ ] Test MATLAB connection
  ```bash
  python test_matlab_setup.py
  ```

- [ ] Test BGN environment
  ```bash
  python test_bgn_environment.py
  ```

- [ ] Verify training works (small test)
  ```python
  # Quick test in Python
  from BGN_MC import BGN_MC
  from stable_baselines3 import TD3
  env = BGN_MC(tmax=1100, pd=True)
  model = TD3('MlpPolicy', env, verbose=1)
  model.learn(total_timesteps=100)  # Small test
  ```

## 🎯 Priority Order

1. **First**: Verify MATLAB setup works (Tasks 5-6)
2. **Second**: Test training pipeline (Task 7)
3. **Third**: Set up quantization testing (Tasks 8-10)
4. **Fourth**: Power profiling (Task 11)
5. **Fifth**: Deployment pipeline (Task 12)

## 📝 Notes

- All scripts now use Windows-compatible encoding (no Unicode symbols)
- MATLAB path is automatically detected from workspace directory
- Test scripts provide clear error messages and troubleshooting tips
- See `README.md` for detailed documentation

## 🔧 Troubleshooting

If you encounter issues:

1. **MATLAB Engine Issues**: See `test_matlab_setup.py` troubleshooting section
2. **Environment Issues**: See `test_bgn_environment.py` troubleshooting section
3. **Import Errors**: Check `requirements.txt` and install missing packages
4. **Path Issues**: All paths are now auto-detected, but verify MATLAB files exist

