# MATLAB Online Integration Instructions

## Step 1: Prepare Simulation Parameters
Run this Python script to generate simulation parameters:
```python
from matlab_online_workflow import MATLABOnlineBridge
bridge = MATLABOnlineBridge()
bridge.prepare_simulation_params(freq=130, amp=2500, pd=True, tmax=1100)
```

## Step 2: Upload to MATLAB Online
1. Go to https://matlab.mathworks.com/
2. Sign in to your MATLAB Online account
3. Upload these files to your MATLAB Online workspace:
   - `matlab_data/simulation_params.json`
   - `matlab_data/run_simulation_online.m`
   - `bgn_init.m` (from your project)
   - `bgn_step.m` (from your project)
   - `gating/` folder (all .m files)

## Step 3: Run Simulation in MATLAB Online
1. In MATLAB Online, run: `run_simulation_online()`
2. Wait for simulation to complete
3. Download `simulation_results.mat` to your local machine

## Step 4: Process Results in Python
```python
results = bridge.load_simulation_results()
if results is not None:
    # Process the results
    sgis = results['sgis']
    terminated = results['terminated']
    print(f"Simulation terminated: {terminated}")
```

## Alternative: Batch Processing
For multiple simulations, you can:
1. Generate multiple parameter files
2. Run them sequentially in MATLAB Online
3. Download all results
4. Process them in batch with Python

## Troubleshooting
- If you get errors, check `simulation_error.mat` for details
- Make sure all required .m files are uploaded
- Verify the gating functions are in the correct path
