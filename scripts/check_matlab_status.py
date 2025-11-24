"""
Check MATLAB Engine installation status
"""

import os
import sys

print("=" * 70)
print("MATLAB ENGINE INSTALLATION STATUS CHECK")
print("=" * 70)

# Check 1: Can we import matlab?
print("\n1. Checking if MATLAB Engine can be imported...")
try:
    import matlab.engine
    print("   [OK] MATLAB Engine module found")
    try:
        version = matlab.engine.__version__
        print(f"   Version: {version}")
    except AttributeError:
        print("   Version: installed")
except ImportError as e:
    print(f"   [X] MATLAB Engine not installed: {e}")
    sys.exit(1)
except RuntimeError as e:
    print(f"   [X] MATLAB Engine found but not properly configured: {e}")
    print("\n   This means the installation is incomplete.")
    print("   The _arch.txt file is missing from the MATLAB directory.")
    print("   You need to run the installer as Administrator.")
    sys.exit(1)

# Check 2: Can we start MATLAB?
print("\n2. Checking if MATLAB can be started...")
try:
    eng = matlab.engine.start_matlab()
    version = eng.version()
    print(f"   [OK] MATLAB started successfully")
    print(f"   MATLAB Version: {version}")
    eng.quit()
except Exception as e:
    print(f"   [X] Failed to start MATLAB: {e}")
    sys.exit(1)

# Check 3: Verify _arch.txt file
print("\n3. Checking _arch.txt file...")
arch_file = r"C:\Program Files\MATLAB\R2025b\extern\engines\python\dist\matlab\engine\_arch.txt"
if os.path.exists(arch_file):
    print(f"   [OK] _arch.txt file exists")
    with open(arch_file, 'r') as f:
        lines = f.readlines()
        print(f"   Architecture: {lines[0].strip()}")
else:
    print(f"   [X] _arch.txt file missing at: {arch_file}")
    print("   This is why the installation failed.")

print("\n" + "=" * 70)
print("STATUS: MATLAB Engine is NOT properly installed")
print("=" * 70)
print("\nTo fix:")
print("1. Right-click 'install_matlab_engine_admin.ps1'")
print("2. Select 'Run with PowerShell'")
print("3. Click 'Yes' for admin access")
print("\nOR run manually as Administrator:")
print('   cd "C:\\Program Files\\MATLAB\\R2025b\\extern\\engines\\python"')
print("   python setup.py install")
print("=" * 70)

