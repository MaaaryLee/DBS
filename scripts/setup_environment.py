"""
Main setup script to verify and configure the environment.
Run this script first to ensure everything is ready.
"""

import sys
import os

def main():
    print("\n" + "=" * 70)
    print("DBS RL Environment Setup Script")
    print("=" * 70)
    print("\nThis script will verify your MATLAB and Python setup.")
    print("Make sure MATLAB is installed and accessible.\n")
    
    # Step 1: Check Python packages
    print("Step 1: Checking Python dependencies...")
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'stable_baselines3': 'Stable-Baselines3',
        'matlab': 'MATLAB Engine',
        'scipy': 'SciPy',
        'gymnasium': 'Gymnasium',
        'antropy': 'Antropy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn'
    }
    
    missing_packages = []
    for module_name, display_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  [OK] {display_name}")
        except ImportError:
            print(f"  [X] {display_name} - MISSING")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    else:
        print("\n[OK] All Python packages are installed")
    
    # Step 2: Test MATLAB connection
    print("\nStep 2: Testing MATLAB engine connection...")
    try:
        from test_matlab_setup import test_matlab_connection
        if not test_matlab_connection():
            print("\n[WARNING] MATLAB setup test failed. Please fix MATLAB issues first.")
            return False
    except Exception as e:
        print(f"\n[X] Could not run MATLAB test: {e}")
        return False
    
    # Step 3: Test BGN environment
    print("\nStep 3: Testing BGN environment...")
    try:
        from test_bgn_environment import test_bgn_environment
        if not test_bgn_environment():
            print("\n[WARNING] BGN environment test failed.")
            return False
    except Exception as e:
        print(f"\n[X] Could not run BGN environment test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Check for model files
    print("\nStep 4: Checking for trained models...")
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_folders = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('TD3_')]
        if model_folders:
            print(f"  [OK] Found {len(model_folders)} model folder(s):")
            for folder in model_folders:
                print(f"    - {folder}")
        else:
            print("  [WARNING] No trained models found. You'll need to train models first.")
    else:
        print("  [WARNING] Models directory doesn't exist. Will be created during training.")
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE [OK]")
    print("=" * 70)
    print("\nYour environment is ready! Next steps:")
    print("1. Train TD3 models: python training.py")
    print("2. Test quantization: See bgnm_testing.ipynb")
    print("3. Run quantization evaluation: python quantize_td3_real.py")
    print("4. Profile power consumption: python power_measure.py")
    print("\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

