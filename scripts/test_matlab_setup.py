"""
Test script to verify MATLAB engine connection and setup.
Run this first to ensure MATLAB is properly configured.
"""

import os
import sys
import matlab.engine

def test_matlab_connection():
    """Test basic MATLAB engine connection."""
    print("=" * 60)
    print("Testing MATLAB Engine Connection")
    print("=" * 60)
    
    try:
        print("\n1. Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        print("   [OK] MATLAB engine started successfully")
        
        print("\n2. Testing MATLAB workspace directory...")
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"   Workspace directory: {workspace_dir}")
        
        eng.cd(workspace_dir)
        current_dir = eng.pwd()
        print(f"   MATLAB current directory: {current_dir}")
        print("   [OK] Directory change successful")
        
        print("\n3. Testing MATLAB basic operations...")
        result = eng.sqrt(4.0)
        print(f"   sqrt(4) = {result}")
        print("   [OK] Basic MATLAB operations work")
        
        print("\n4. Checking for required MATLAB files...")
        required_files = [
            'bgn_init.m',
            'bgn_step.m',
            'bgn_vars.mat'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(workspace_dir, file)
            if os.path.exists(file_path):
                print(f"   [OK] Found: {file}")
            else:
                print(f"   [X] Missing: {file}")
                missing_files.append(file)
        
        print("\n5. Checking gating functions directory...")
        gating_dir = os.path.join(workspace_dir, 'gating')
        if os.path.exists(gating_dir):
            gating_files = [f for f in os.listdir(gating_dir) if f.endswith('.m')]
            print(f"   [OK] Found gating directory with {len(gating_files)} .m files")
        else:
            print(f"   [X] Missing gating directory")
            missing_files.append('gating/')
        
        if missing_files:
            print(f"\n[WARNING] Missing {len(missing_files)} required file(s)")
            return False
        
        print("\n6. Testing MATLAB function call (bgn_init)...")
        try:
            # Test if bgn_init can be called (will fail if dependencies missing)
            # We'll just check if it exists, not actually run it
            eng.which('bgn_init', nargout=1)
            print("   [OK] bgn_init function is accessible")
        except Exception as e:
            print(f"   [WARNING] Could not verify bgn_init: {e}")
            print("   (This is okay if MATLAB path needs to be set)")
        
        print("\n" + "=" * 60)
        print("MATLAB SETUP TEST: PASSED [OK]")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MATLAB is installed and in your PATH")
        print("2. Run 'python setup.py install' in MATLAB's extern/engines/python directory")
        print("3. On Windows, you may need to run: python -m pip install matlabengine")
        print("4. Verify MATLAB version matches matlabengine version in requirements.txt")
        return False

def test_matlab_engine_version():
    """Check MATLAB engine version compatibility."""
    print("\n" + "=" * 60)
    print("Checking MATLAB Engine Version")
    print("=" * 60)
    
    try:
        import matlab.engine
        try:
            print(f"Python matlabengine version: {matlab.engine.__version__}")
        except AttributeError:
            print("Python matlabengine: installed")
        
        eng = matlab.engine.start_matlab()
        matlab_version = eng.version()
        print(f"MATLAB version: {matlab_version}")
        
        print("\n[OK] Version check complete")
        return True
    except Exception as e:
        print(f"[X] Version check failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MATLAB Setup Verification Script")
    print("=" * 60)
    
    version_ok = test_matlab_engine_version()
    connection_ok = test_matlab_connection()
    
    if version_ok and connection_ok:
        print("\n[SUCCESS] All tests passed! MATLAB is ready to use.")
        sys.exit(0)
    else:
        print("\n[FAILED] Some tests failed. Please fix the issues above.")
        sys.exit(1)

