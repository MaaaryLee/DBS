"""
Test script to verify BGN environment works with MATLAB engine.
Run this after test_matlab_setup.py passes.
"""

import numpy as np
from BGN_MC import BGN_MC

def test_bgn_environment():
    """Test BGN environment initialization and basic operations."""
    print("=" * 60)
    print("Testing BGN Environment")
    print("=" * 60)
    
    try:
        print("\n1. Creating BGN environment (Parkinsonian state)...")
        env = BGN_MC(tmax=1100, pd=True, mode='hvgi_sgi')
        print("   [OK] Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        print("\n2. Testing environment reset...")
        obs, info = env.reset()
        print(f"   [OK] Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation values: {obs}")
        print(f"   Info keys: {info.keys()}")
        
        print("\n3. Testing environment step (no action)...")
        obs, reward, terminated, truncated, info = env.step()
        print(f"   [OK] Step successful")
        print(f"   Reward: {reward:.4f}")
        print(f"   Terminated: {terminated}")
        print(f"   Info: {info}")
        
        print("\n4. Testing environment step (with DBS action)...")
        # Normalized action: [frequency, amplitude]
        # freq = 130 Hz -> normalized: (130/185)*2 - 1 = 0.405
        # amp = 2500 mA -> normalized: (2500/5000)*2 - 1 = 0.0
        action = np.array([0.405, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   [OK] Step with action successful")
        print(f"   Action: {action}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Observation: {obs}")
        
        print("\n5. Testing full episode simulation...")
        obs, info = env.reset()
        step_count = 0
        total_reward = 0
        
        while not terminated and step_count < 5:  # Limit to 5 steps for testing
            action = np.array([0.0, 0.0])  # No DBS
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        print(f"   [OK] Episode simulation successful")
        print(f"   Steps completed: {step_count}")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Final terminated: {terminated}")
        
        print("\n6. Testing different observation modes...")
        modes = ['hvgi', 'hsgi', 'hvgi_sgi']
        for mode in modes:
            try:
                env_test = BGN_MC(tmax=1100, pd=True, mode=mode)
                obs_test, _ = env_test.reset()
                print(f"   [OK] Mode '{mode}': observation shape {obs_test.shape}")
            except Exception as e:
                print(f"   [X] Mode '{mode}': {e}")
        
        print("\n" + "=" * 60)
        print("BGN ENVIRONMENT TEST: PASSED [OK]")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure test_matlab_setup.py passed first")
        print("2. Check that bgn_init.m and bgn_step.m are in the workspace")
        print("3. Verify gating functions are accessible")
        print("4. Check MATLAB console for any error messages")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BGN Environment Verification Script")
    print("=" * 60)
    print("\nMake sure test_matlab_setup.py passed before running this.\n")
    
    success = test_bgn_environment()
    
    if success:
        print("\n[SUCCESS] BGN environment is working correctly!")
        print("You can now proceed with training and quantization experiments.")
    else:
        print("\n[FAILED] BGN environment test failed. Please fix the issues above.")

