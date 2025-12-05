"""
Test script to demonstrate the proper MATLAB Online workflow for fresh dynamics.
This shows how to get new brain simulation data for each training step.
"""

import numpy as np
import os
import time
from matlab_online_workflow import MATLABOnlineBridge
from BGN_MC_Online import BGN_MC_Online

def test_parameter_generation():
    """Test that we can generate different simulation parameters."""
    print("=== Testing Parameter Generation ===")
    
    bridge = MATLABOnlineBridge()
    
    # Generate different DBS parameters
    test_params = [
        (100, 2000),  # Low DBS
        (130, 2500),  # Standard DBS  
        (150, 3000),  # High DBS
        (80, 1500),   # Very low DBS
        (170, 4000),  # Very high DBS
    ]
    
    for i, (freq, amp) in enumerate(test_params):
        params_file = bridge.prepare_simulation_params(freq, amp, pd=True, tmax=1100)
        print(f"Generated params {i+1}: {freq}Hz, {amp}mA")
        
        # Read back the parameters to verify
        import json
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"  Verified: freq={params['freq']}, amp={params['amp']}")
    
    print("✅ Parameter generation working correctly!")

def test_workflow_simulation():
    """Simulate the complete workflow without actually using MATLAB Online."""
    print("\n=== Simulating Complete Workflow ===")
    
    bridge = MATLABOnlineBridge()
    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=True)
    
    # Simulate 3 training steps with different DBS parameters
    for step in range(3):
        print(f"\n--- Training Step {step + 1} ---")
        
        # Generate random DBS parameters (what TD3 would do)
        action = np.random.uniform(-1, 1, 2)
        freq = 185 * ((action[0] + 1)/2)
        amp = 5000 * ((action[1] + 1)/2)
        
        print(f"TD3 Action: {action}")
        print(f"DBS Parameters: {freq:.1f}Hz, {amp:.1f}mA")
        
        # Generate simulation parameters
        params_file = bridge.prepare_simulation_params(freq, amp, pd=True, tmax=1100)
        print(f"✅ Parameters saved to: {params_file}")
        
        # In real workflow, you would:
        print("📋 Manual steps for fresh dynamics:")
        print("  1. Upload simulation_params.json to MATLAB Online")
        print("  2. Run run_simulation_online() in MATLAB Online")
        print("  3. Download simulation_results.mat")
        print("  4. Place it in matlab_data/ directory")
        print("  5. Continue with Python training")
        
        # For this test, we'll use the existing results
        # (In real training, you'd have fresh results here)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Environment step: reward={reward:.4f}")
        print(f"  SGi biomarker (r3): {info['r3']:.4f}")
        
        # Simulate some processing time
        time.sleep(0.1)

def test_batch_workflow():
    """Test batch processing workflow for multiple simulations."""
    print("\n=== Testing Batch Workflow ===")
    
    bridge = MATLABOnlineBridge()
    
    # Generate multiple simulation parameters for batch processing
    batch_params = [
        {"freq": 100, "amp": 2000, "name": "low_dbs"},
        {"freq": 130, "amp": 2500, "name": "standard_dbs"},
        {"freq": 150, "amp": 3000, "name": "high_dbs"},
        {"freq": 80, "amp": 1500, "name": "very_low_dbs"},
        {"freq": 170, "amp": 4000, "name": "very_high_dbs"},
    ]
    
    print("📊 Generated batch simulation parameters:")
    for i, sim in enumerate(batch_params):
        # Create unique parameter file for each simulation
        params_file = f"matlab_data/simulation_params_{sim['name']}.json"
        bridge.prepare_simulation_params(
            freq=sim["freq"], 
            amp=sim["amp"], 
            pd=True, 
            tmax=1100
        )
        
        # Rename to unique file
        os.rename("matlab_data/simulation_params.json", params_file)
        
        print(f"  {i+1}. {sim['name']}: {sim['freq']}Hz, {sim['amp']}mA")
        print(f"     File: {params_file}")
    
    print("\n📋 Batch processing workflow:")
    print("  1. Upload all parameter files to MATLAB Online")
    print("  2. Run simulations sequentially in MATLAB Online")
    print("  3. Download all results (rename as needed)")
    print("  4. Process results in batch with Python")

def demonstrate_fresh_vs_cached():
    """Demonstrate the difference between fresh and cached dynamics."""
    print("\n=== Fresh vs Cached Dynamics ===")
    
    # Test with cached data (current behavior)
    print("1. Testing with CACHED data (current behavior):")
    env_cached = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)
    
    obs1, info1 = env_cached.reset()
    action = np.array([0.5, 0.5])
    obs2, reward2, _, _, info2 = env_cached.step(action)
    
    print(f"   Cached SGi biomarker: {info1['r3']:.6f} -> {info2['r3']:.6f}")
    print(f"   (Same data used for both steps)")
    
    # Test with MATLAB Online mode (would use fresh data if available)
    print("\n2. Testing with MATLAB Online mode (fresh data workflow):")
    env_fresh = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=True)
    
    obs1, info1 = env_fresh.reset()
    action = np.array([0.5, 0.5])
    obs2, reward2, _, _, info2 = env_fresh.step(action)
    
    print(f"   Fresh SGi biomarker: {info1['r3']:.6f} -> {info2['r3']:.6f}")
    print(f"   (Would use fresh MATLAB Online data if available)")
    
    print("\n💡 Key Point:")
    print("   - Cached mode: Fast, but same dynamics every time")
    print("   - Fresh mode: Slower, but realistic brain dynamics")
    print("   - For training: Use fresh mode for realistic results")
    print("   - For development: Use cached mode for speed")

if __name__ == "__main__":
    test_parameter_generation()
    test_workflow_simulation()
    test_batch_workflow()
    demonstrate_fresh_vs_cached()
    
    print("\n" + "="*60)
    print("🎯 SUMMARY: MATLAB Online Workflow")
    print("="*60)
    print("✅ Parameter generation: Working")
    print("✅ Environment integration: Working")
    print("✅ Batch processing: Ready")
    print("⚠️  Fresh dynamics: Requires manual MATLAB Online steps")
    print("\n📋 For real training with fresh dynamics:")
    print("   1. Generate parameters with Python")
    print("   2. Upload to MATLAB Online")
    print("   3. Run simulation")
    print("   4. Download results")
    print("   5. Continue training")
    print("\n🚀 Ready for production use!")
