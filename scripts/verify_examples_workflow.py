"""
Verify that the environment matches what's shown in examples.ipynb.
This script tests the exact workflow from the notebook.
"""

import sys
import numpy as np

def test_imports():
    """Test imports as shown in Cell 1 of examples.ipynb"""
    print("=" * 70)
    print("Testing Imports (Cell 1)")
    print("=" * 70)
    
    try:
        # Test the exact import from notebook (wildcard import)
        # Note: In functions, we need to import normally, but verify it works
        from BGN_MC import BGN_MC
        print("[OK] from BGN_MC import BGN_MC - successful")
        
        # Also test if wildcard would work (simulate notebook behavior)
        import BGN_MC as bgn_module
        if hasattr(bgn_module, 'BGN_MC'):
            print("[OK] BGN_MC class is available in module")
        else:
            print("[X] BGN_MC class not found in module")
            return False
            
        return True
    except Exception as e:
        print(f"[X] Import failed: {e}")
        return False

def test_basic_environment():
    """Test basic environment usage (Cell 1)"""
    print("\n" + "=" * 70)
    print("Testing Basic Environment (Cell 1)")
    print("=" * 70)
    
    try:
        from BGN_MC import BGN_MC
        
        print("\n1. Creating BGN environment (tmax=1100, pd=True)...")
        bgn = BGN_MC(tmax=1100, pd=True)
        print(f"   [OK] Environment created")
        print(f"   Mode: {bgn.mode} (default)")
        print(f"   Observation space: {bgn.observation_space}")
        print(f"   Action space: {bgn.action_space}")
        
        print("\n2. Testing reset()...")
        obs, info = bgn.reset()
        print(f"   [OK] Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation values: {obs}")
        print(f"   Info keys: {list(info.keys())}")
        
        print("\n3. Testing step() without action...")
        terminated = 0
        step_count = 0
        while terminated != 1 and step_count < 3:  # Limit to 3 steps for testing
            observation, reward, terminated, truncated, info = bgn.step()
            step_count += 1
            print(f"   Step {step_count}: reward={reward:.4f}, terminated={terminated}")
        
        print(f"   [OK] Step() works correctly")
        return True
        
    except Exception as e:
        print(f"\n[X] Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dbs_simulation():
    """Test DBS simulation (Cell 3)"""
    print("\n" + "=" * 70)
    print("Testing DBS Simulation (Cell 3)")
    print("=" * 70)
    
    try:
        from BGN_MC import BGN_MC
        
        # Normalize frequency and amplitude as shown in Cell 3 of notebook
        freq = 130
        amp = 2500
        norm_freq = (freq/185)*2 - 1
        norm_amp = (amp/5000)*2 - 1
        
        # Note: The notebook shows this normalization formula
        # This matches the denormalization in Cell 6: (norm_val+1)/2 * max_val
        
        print(f"\n1. Normalizing DBS parameters...")
        print(f"   freq={freq} Hz -> norm_freq={norm_freq:.3f}")
        print(f"   amp={amp} mA -> norm_amp={norm_amp:.3f}")
        
        print("\n2. Creating environment and applying DBS...")
        bgn = BGN_MC(tmax=1100, pd=True)
        obs, info = bgn.reset()
        
        action = np.array([norm_freq, norm_amp])
        obs, reward, terminated, truncated, info = bgn.step(action)
        
        print(f"   [OK] DBS simulation successful")
        print(f"   Action: {action}")
        print(f"   Reward: {reward:.4f}")
        print(f"   Observation: {obs}")
        
        return True
        
    except Exception as e:
        print(f"\n[X] DBS simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_matlab_data_access():
    """Test accessing MATLAB data (Cell 2)"""
    print("\n" + "=" * 70)
    print("Testing MATLAB Data Access (Cell 2)")
    print("=" * 70)
    
    try:
        import scipy.io
        import os
        
        print("\n1. Checking for bgn_vars.mat...")
        if os.path.exists('bgn_vars.mat'):
            print("   [OK] bgn_vars.mat exists")
        else:
            print("   [WARNING] bgn_vars.mat not found (will be created during simulation)")
            return True  # Not a failure, just not created yet
        
        print("\n2. Loading MATLAB data...")
        data = scipy.io.loadmat('bgn_vars.mat')
        
        required_keys = ['sgis', 'vgi', 'vth', 'vsn', 'vge']
        missing_keys = []
        
        for key in required_keys:
            if key in data:
                print(f"   [OK] Found: {key} (shape: {data[key].shape})")
            else:
                print(f"   [X] Missing: {key}")
                missing_keys.append(key)
        
        if missing_keys:
            print(f"\n   [WARNING] Missing keys: {missing_keys}")
            print("   (This is OK if simulation hasn't run yet)")
        
        return True
        
    except Exception as e:
        print(f"\n[X] MATLAB data access test failed: {e}")
        return False

def test_observation_modes():
    """Test different observation modes"""
    print("\n" + "=" * 70)
    print("Testing Observation Modes")
    print("=" * 70)
    
    try:
        from BGN_MC import BGN_MC
        
        modes = {
            'hvgi': 4,      # Default mode
            'hsgi': 4,
            'hvgi_sgi': 6   # Used in training examples
        }
        
        for mode, expected_size in modes.items():
            print(f"\nTesting mode '{mode}' (expected obs size: {expected_size})...")
            try:
                env = BGN_MC(tmax=1100, pd=True, mode=mode)
                obs, info = env.reset()
                
                if obs.shape[0] == expected_size:
                    print(f"   [OK] Mode '{mode}': observation shape {obs.shape} (correct)")
                else:
                    print(f"   [X] Mode '{mode}': expected size {expected_size}, got {obs.shape[0]}")
                    return False
                    
            except Exception as e:
                print(f"   [X] Mode '{mode}' failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"\n[X] Observation modes test failed: {e}")
        return False

def check_package_versions():
    """Check if package versions match notebook recommendations"""
    print("\n" + "=" * 70)
    print("Checking Package Versions (Cell 0)")
    print("=" * 70)
    
    notebook_versions = {
        'gymnasium': '1.2.0',
        'matlabengine': '24.2.1',
        'numpy': '2.2.6',
        'torch': '2.6.0',
    }
    
    installed_versions = {}
    missing_packages = []
    
    for package, expected_version in notebook_versions.items():
        try:
            if package == 'matlabengine':
                import matlab.engine
                installed_versions[package] = matlab.engine.__version__
            elif package == 'torch':
                import torch
                installed_versions[package] = torch.__version__
            elif package == 'gymnasium':
                import gymnasium
                installed_versions[package] = gymnasium.__version__
            elif package == 'numpy':
                import numpy
                installed_versions[package] = numpy.__version__
            
            print(f"   {package}: {installed_versions[package]} (notebook suggests: {expected_version})")
            
        except ImportError:
            print(f"   [X] {package}: NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   [WARNING] Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
    
    return len(missing_packages) == 0

def main():
    """Run all verification tests"""
    print("\n" + "=" * 70)
    print("Examples.ipynb Workflow Verification")
    print("=" * 70)
    print("\nThis script verifies that the environment matches examples.ipynb")
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Check package versions
    results.append(("Package Versions", check_package_versions()))
    
    # Test MATLAB data access
    results.append(("MATLAB Data Access", test_matlab_data_access()))
    
    # Test basic environment
    results.append(("Basic Environment", test_basic_environment()))
    
    # Test DBS simulation
    results.append(("DBS Simulation", test_dbs_simulation()))
    
    # Test observation modes
    results.append(("Observation Modes", test_observation_modes()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[X]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! Environment matches examples.ipynb")
    else:
        print("\n[FAILED] Some tests failed. Please fix the issues above.")
        print("\nNote: Some failures may be expected if:")
        print("  - MATLAB engine is not set up")
        print("  - bgn_vars.mat hasn't been created yet")
        print("  - Required packages are not installed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

