"""
Example: Using MATLAB Online with the DBS project

This example shows how to:
1. Use the MATLAB Online environment for training
2. Trigger MATLAB Online simulations
3. Process results from MATLAB Online
"""

import numpy as np
from BGN_MC_Online import BGN_MC_Online
from matlab_online_workflow import MATLABOnlineBridge



def example_basic_usage():
    """Basic usage example with pre-computed data."""
    print("=== Basic Usage Example ===")
    
    # Create environment (uses pre-computed data by default)
    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    # Run a few steps
    total_reward = 0
    for i in range(5):
        # Try different DBS parameters
        if i < 2:
            action = np.array([0.0, 0.0])  # No DBS
        elif i < 4:
            action = np.array([0.3, 0.3])  # Low DBS
        else:
            action = np.array([0.7, 0.7])  # High DBS
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        freq = 185 * ((action[0] + 1)/2)
        amp = 5000 * ((action[1] + 1)/2)
        
        print(f"Step {i+1}: DBS({freq:.1f}Hz, {amp:.1f}mA) -> reward={reward:.4f}")
        
        if terminated:
            break
    
    print(f"Total reward: {total_reward:.4f}")

def example_matlab_online_workflow():
    """Example of using MATLAB Online workflow."""
    print("\n=== MATLAB Online Workflow Example ===")
    
    # Create environment with MATLAB Online enabled
    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=True)
    
    # Test different DBS parameters
    test_params = [
        (130, 2500),  # Traditional DBS
        (100, 2000),  # Lower frequency
        (150, 3000),  # Higher amplitude
    ]
    
    for freq, amp in test_params:
        print(f"\n🧠 Testing DBS: {freq}Hz, {amp}mA")
        
        # Trigger MATLAB Online simulation
        success = env.trigger_matlab_online_simulation(freq, amp)
        
        if success:
            print("📋 Manual steps required:")
            print("1. Go to MATLAB Online")
            print("2. Upload matlab_data/simulation_params.json")
            print("3. Run the simulation")
            print("4. Download results")
            print("5. Place simulation_results.mat in matlab_data/")
            print("6. Run this script again to process results")
            break  # Only show for first example

def example_batch_processing():
    """Example of batch processing multiple simulations."""
    print("\n=== Batch Processing Example ===")
    
    bridge = MATLABOnlineBridge()
    
    # Generate multiple simulation parameters
    simulations = [
        {"freq": 100, "amp": 2000, "name": "low_dbs"},
        {"freq": 130, "amp": 2500, "name": "standard_dbs"},
        {"freq": 150, "amp": 3000, "name": "high_dbs"},
    ]
    
    print("📊 Generated simulation parameters:")
    for sim in simulations:
        params_file = bridge.prepare_simulation_params(
            freq=sim["freq"], 
            amp=sim["amp"], 
            pd=True, 
            tmax=1100
        )
        print(f"  {sim['name']}: {sim['freq']}Hz, {sim['amp']}mA")
    
    print("\n📋 Next steps for batch processing:")
    print("1. Upload all parameter files to MATLAB Online")
    print("2. Run simulations sequentially")
    print("3. Download all results")
    print("4. Process with Python")

def example_training_integration():
    """Example of how to integrate with training."""
    print("\n=== Training Integration Example ===")
    
    # This shows how you could modify the training script
    print("To integrate with training, modify training.py:")
    print("""
# In training.py, replace:
# from BGN_MC import BGN_MC
# env = BGN_MC(tmax=1100)

# With:
from BGN_MC_Online import BGN_MC_Online
env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=True)

# The rest of the training code remains the same!
""")

if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_matlab_online_workflow()
    example_batch_processing()
    example_training_integration()
    
    print("\n✅ All examples completed!")
    print("\n📚 For detailed instructions, see: matlab_data/MATLAB_Online_Instructions.md")

from matlab_online_workflow import wait_for_json_change, print_matlab_reminder

def watch_and_remind(json_path: str = "matlab_data/simulation_params.json"):
    last = None
    print("Watching for new simulation_params.json updates… Press Ctrl+C to stop")
    while True:
        changed = wait_for_json_change(json_path=json_path, last_seen_mtime=last, poll_seconds=1.0)
        if changed is not None:
            last = changed
            print_matlab_reminder(json_path=json_path)

from typing import List, Tuple
from scipy.io import loadmat
import numpy as np
from matlab_online_workflow import list_mat_results

def iterate_mat_folder(
    folder: str = "matlab_data",
    prefix: str = "simulation_results",
    sort_by: str = "name",
) -> List[str]:
    files = list_mat_results(folder=folder, prefix=prefix, sort_by=sort_by)
    if not files:
        print("No MAT files found in", folder)
    else:
        print("Batch files:", files)
    return files

def load_sgis_and_meta(mat_path: str) -> Tuple[np.ndarray, dict]:
    data = loadmat(mat_path)
    sgis = np.asarray(data.get("sgis")).squeeze()
    meta = {k: data[k] for k in ("freq", "amp", "pd", "tmax") if k in data}
    return sgis, meta

def demo_train_over_batch(env, folder: str = "matlab_data", prefix: str = "simulation_results"):
    files = iterate_mat_folder(folder=folder, prefix=prefix, sort_by="name")
    for i, f in enumerate(files, start=1):
        sgis, meta = load_sgis_and_meta(f)
        # Example of feeding data into your env
        # If your env already knows how to read the MAT on its own, just set the path and step
        if hasattr(env, "set_external_mat_path"):
            env.set_external_mat_path(f)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"Step {i} using {f}")
        print("   freq", float(meta.get("freq", [np.nan])[0] if "freq" in meta else np.nan))
        print("   amp", float(meta.get("amp", [np.nan])[0] if "amp" in meta else np.nan))
        print("   reward", float(reward))
        if terminated or truncated:
            env.reset()
def operator_checklist():
    print("Operator checklist")
    print("1. Upload simulation_params.json to MATLAB Online inside matlab_data")
    print("2. In MATLAB run: run_simulation_online")
    print("3. Download simulation_results.mat into matlab_data locally")
    print("4. Continue training in Python")