"""
MATLAB Online Integration Workflow for DBS Project

This script provides a bridge between your local Python environment and MATLAB Online.
Since MATLAB Online doesn't support direct Python integration, we use a file-based approach.

Workflow:
1. Python generates simulation parameters and saves them to files
2. You upload these files to MATLAB Online and run the simulation
3. MATLAB Online saves results to files
4. You download the results and Python processes them
"""

import numpy as np
import json
import os
from datetime import datetime

class MATLABOnlineBridge:
    def __init__(self, data_dir="matlab_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def prepare_simulation_params(self, freq, amp, pd=True, tmax=1100):
        """Prepare simulation parameters for MATLAB Online."""
        params = {
            'freq': float(freq),
            'amp': float(amp),
            'pd': int(pd),
            'tmax': int(tmax),
            'timestamp': datetime.now().isoformat(),
            'sim_time': 10000  # Default simulation time
        }
        
        # Save parameters to JSON file
        params_file = os.path.join(self.data_dir, 'simulation_params.json')
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
            
        print(f"✅ Simulation parameters saved to: {params_file}")
        print(f"Parameters: freq={freq}Hz, amp={amp}mA, pd={pd}, tmax={tmax}ms")
        
        return params_file
    
    def create_matlab_script(self, output_file="run_simulation_online.m"):
        """Create a MATLAB script that can be run in MATLAB Online."""
        matlab_script = '''function run_simulation_online()
    % MATLAB Online script for DBS simulation
    % This script reads parameters from simulation_params.json and runs the simulation
    
    try
        % Read simulation parameters
        params_file = 'simulation_params.json';
        if ~exist(params_file, 'file')
            error('simulation_params.json not found. Please upload it first.');
        end
        
        % Read JSON parameters (requires JSON toolbox or manual parsing)
        fid = fopen(params_file, 'r');
        raw = fread(fid, inf);
        str = char(raw');
        fclose(fid);
        
        % Parse JSON manually (simple approach)
        freq = extract_value(str, '"freq"');
        amp = extract_value(str, '"amp"');
        pd = extract_value(str, '"pd"');
        tmax = extract_value(str, '"tmax"');
        sim_time = extract_value(str, '"sim_time"');
        
        fprintf('Running simulation with: freq=%.1fHz, amp=%.1fmA, pd=%d, tmax=%dms\\n', ...
                freq, amp, pd, tmax);
        
        % Initialize simulation
        bgn_init(pd, tmax);
        
        % Run simulation step
        [terminated, sgis] = bgn_step(freq, amp, sim_time);
        
        % Save results
        save('simulation_results.mat', 'terminated', 'sgis', 'freq', 'amp', 'pd', 'tmax');
        
        fprintf('✅ Simulation completed. Results saved to simulation_results.mat\\n');
        fprintf('Terminated: %d\\n', terminated);
        
    catch ME
        fprintf('❌ Error: %s\\n', ME.message);
        % Save error information
        error_info = struct('error', ME.message, 'stack', ME.stack);
        save('simulation_error.mat', 'error_info');
    end
end

function value = extract_value(json_str, field_name)
    % Simple JSON value extraction
    pattern = [field_name, '\\s*:\\s*([^,}\\]]+)'];
    match = regexp(json_str, pattern, 'tokens');
    if ~isempty(match)
        value_str = strtrim(match{1}{1});
        % Remove quotes if present
        if value_str(1) == '"' && value_str(end) == '"'
            value_str = value_str(2:end-1);
        end
        value = str2double(value_str);
        if isnan(value)
            value = value_str;
        end
    else
        value = [];
    end
end
'''
        
        script_path = os.path.join(self.data_dir, output_file)
        with open(script_path, 'w') as f:
            f.write(matlab_script)
            
        print(f"✅ MATLAB script created: {script_path}")
        return script_path
    
    def load_simulation_results(self, results_file="simulation_results.mat"):
        """Load simulation results from MATLAB Online."""
        import scipy.io
        
        results_path = os.path.join(self.data_dir, results_file)
        
        if not os.path.exists(results_path):
            print(f"❌ Results file not found: {results_path}")
            print("Please download simulation_results.mat from MATLAB Online first.")
            return None
            
        try:
            results = scipy.io.loadmat(results_path)
            print(f"✅ Results loaded from: {results_path}")
            return results
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return None
    
    def create_instructions_file(self):
        """Create step-by-step instructions for using MATLAB Online."""
        instructions = """# MATLAB Online Integration Instructions

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
"""
        
        instructions_path = os.path.join(self.data_dir, "MATLAB_Online_Instructions.md")
        with open(instructions_path, 'w') as f:
            f.write(instructions)
            
        print(f"✅ Instructions created: {instructions_path}")
        return instructions_path

def test_workflow():
    """Test the MATLAB Online workflow."""
    print("=== Testing MATLAB Online Workflow ===")
    
    bridge = MATLABOnlineBridge()
    
    # Test parameter generation
    params_file = bridge.prepare_simulation_params(freq=130, amp=2500, pd=True)
    
    # Create MATLAB script
    script_file = bridge.create_matlab_script()
    
    # Create instructions
    instructions_file = bridge.create_instructions_file()
    
    print("\n✅ Workflow setup complete!")
    print(f"📁 Data directory: {bridge.data_dir}")
    print(f"📄 Parameters: {params_file}")
    print(f"📄 MATLAB script: {script_file}")
    print(f"📄 Instructions: {instructions_file}")
    
    print("\n📋 Next steps:")
    print("1. Upload the files to MATLAB Online")
    print("2. Run the simulation")
    print("3. Download results")
    print("4. Process with Python")

if __name__ == "__main__":
    test_workflow()

# ===== MATLAB Online integration helpers =====
import os
import time
from typing import Iterable, List, Optional
import glob

def get_sim_json_mtime(json_path: str = "matlab_data/simulation_params.json") -> Optional[float]:
    try:
        return os.path.getmtime(json_path)
    except FileNotFoundError:
        return None

def wait_for_json_change(
    json_path: str = "matlab_data/simulation_params.json",
    poll_seconds: float = 2.0,
    last_seen_mtime: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
) -> Optional[float]:
    start = time.time()
    if last_seen_mtime is None:
        last_seen_mtime = get_sim_json_mtime(json_path)
    while True:
        current = get_sim_json_mtime(json_path)
        if current is not None and current != last_seen_mtime:
            return current
        if timeout_seconds is not None and (time.time() - start) >= timeout_seconds:
            return None
        time.sleep(poll_seconds)

def print_matlab_reminder(json_path: str = "matlab_data/simulation_params.json") -> None:
    print("New parameters detected in", json_path)
    print("1. Upload simulation_params.json to MATLAB Online inside matlab_data")
    print("2. In MATLAB run: run_simulation_online")
    print("3. Download simulation_results.mat back into matlab_data locally")
    print("4. Continue training in Python")

def list_mat_results(
    folder: str = "matlab_data",
    prefix: str = "simulation_results",
    allowed_suffixes: Iterable[str] = (".mat",),
    sort_by: str = "name",  # 'name' or 'mtime'
    reverse: bool = False,
) -> List[str]:
    paths: List[str] = []
    for ext in allowed_suffixes:
        paths.extend(glob.glob(os.path.join(folder, f"{prefix}*{ext}")))
    if sort_by == "mtime":
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=reverse)
    else:
        paths.sort(reverse=reverse)
    return paths