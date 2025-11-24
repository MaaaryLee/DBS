# MATLAB Engine Usage Explanation

## When MATLAB Engine is Used

### 1. **Power Profiling Script (`power_profile_windows.py`)**

**MATLAB is used ONLY during model loading:**
- Line 154: `env = BGN_MC(tmax=1100, pd=True)` - Creates environment (starts MATLAB)
- This happens once when loading the model
- **NOT used during the actual profiling loop** - just runs inference on pre-saved states

**The profiling loop (`run_inference()`):**
- Lines 102-107: Just runs `policy(x)` on saved states
- No `env.step()` calls = No MATLAB calls
- Pure PyTorch inference only

**Why this matters:**
- MATLAB overhead doesn't affect power profiling measurements
- We're measuring pure model inference, not environment simulation

### 2. **Comprehensive Evaluation Script (`comprehensive_quantization_eval.py`)**

**MATLAB is used during episode evaluation:**
- Line 182: `observation = env.reset()[0]` - Calls `eng.bgn_init()` in MATLAB
- Line 198: `env.step(action)` - Calls `eng.bgn_step()` in MATLAB
- Line 203: `scipy.io.loadmat('bgn_vars.mat')` - Reads MATLAB output file

**This happens in `evaluate_performance()` function:**
- Runs 5 episodes for each model (FP32 and INT8)
- Each episode calls MATLAB multiple times to simulate brain dynamics
- This is where MATLAB does the heavy computation!

## Why No Graphs Were Shown

### Power Profiling Script:
- **No graphs created** - only saves JSON results
- Just measures CPU usage and inference time

### Comprehensive Evaluation Script:
- **Graphs ARE created** but saved to files, not displayed:
  - Line 274: `plt.savefig(f'{output_dir}/model_size_comparison.png')`
  - Line 275: `plt.close()` - Closes figure (doesn't show it)
  - Graphs saved to `quantization_eval_plots/` directory

**To see the graphs:**
- Check the `quantization_eval_plots/` folder
- Files: `model_size_comparison.png`, `inference_time_comparison.png`, `action_difference_histogram.png`

## Summary

| Script | MATLAB Usage | Graphs |
|--------|-------------|--------|
| **Power Profiling** | Only during model loading (minimal) | None created |
| **Comprehensive Eval** | During episode evaluation (heavy usage) | Created but saved to files, not displayed |

## Where MATLAB Actually Does Work

**During `evaluate_performance()` in comprehensive evaluation:**
1. `env.reset()` → Calls `eng.bgn_init()` - Initializes brain model in MATLAB
2. `env.step(action)` → Calls `eng.bgn_step()` - Simulates 100ms of brain dynamics
3. Reads `bgn_vars.mat` - Gets brain state from MATLAB

**This is where MATLAB computes:**
- Neuron dynamics
- Synaptic connections
- Brain oscillations (beta waves, etc.)
- DBS stimulation effects

The power profiling script skips all this - it just measures pure model inference speed!

