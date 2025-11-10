"""
Quick setup check - shows what's installed and what's missing
"""

print("=" * 70)
print("SETUP STATUS CHECK")
print("=" * 70)

packages = {
    'gymnasium': 'Gymnasium (RL environments)',
    'numpy': 'NumPy',
    'torch': 'PyTorch',
    'stable_baselines3': 'Stable-Baselines3 (RL algorithms)',
    'scipy': 'SciPy',
    'matplotlib': 'Matplotlib',
    'sklearn': 'scikit-learn',
    'antropy': 'Antropy',
    'matlab': 'MATLAB Engine'
}

print("\nPackage Status:")
print("-" * 70)

all_ok = True
for module_name, display_name in packages.items():
    try:
        if module_name == 'matlab':
            import matlab.engine
            try:
                version = matlab.engine.__version__
            except AttributeError:
                version = "installed"
            print(f"  [OK] {display_name:40s} {version}")
        elif module_name == 'torch':
            import torch
            print(f"  [OK] {display_name:40s} v{torch.__version__}")
        elif module_name == 'numpy':
            import numpy
            print(f"  [OK] {display_name:40s} v{numpy.__version__}")
        elif module_name == 'gymnasium':
            import gymnasium
            print(f"  [OK] {display_name:40s} v{gymnasium.__version__}")
        else:
            __import__(module_name)
            print(f"  [OK] {display_name:40s} installed")
    except ImportError:
        print(f"  [X] {display_name:40s} MISSING")
        all_ok = False

print("\n" + "=" * 70)
if all_ok:
    print("[SUCCESS] All packages installed!")
    print("\nNext: Test the environment with: python test_cell1.py")
else:
    print("[ACTION NEEDED] Some packages are missing")
    if 'matlab' in [p for p, d in packages.items()]:
        print("\nTo install MATLAB Engine:")
        print("  1. Right-click 'install_matlab_engine.bat'")
        print("  2. Select 'Run as administrator'")
        print("  OR run manually:")
        print('     cd "C:\\Program Files\\MATLAB\\R2025b\\extern\\engines\\python"')
        print("     python setup.py install")
print("=" * 70)

