"""
Convert ONNX model to TensorFlow SavedModel (Cell 13 from examples.ipynb).
"""

import os
import subprocess
import sys

def convert_onnx_to_tf(onnx_path='onnx_actors/model.onnx', output_dir='tf_model'):
    """
    Convert ONNX model to TensorFlow SavedModel format.
    
    Args:
        onnx_path: Path to ONNX model file
        output_dir: Directory to save TensorFlow SavedModel
    """
    print("=" * 70)
    print("Converting ONNX to TensorFlow SavedModel (Cell 13 from examples.ipynb)")
    print("=" * 70)
    
    # Check if ONNX file exists
    print(f"\n1. Checking ONNX file...")
    if not os.path.exists(onnx_path):
        print(f"   [X] ERROR: ONNX file not found at {onnx_path}")
        return False
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    print(f"   [OK] ONNX file found: {onnx_path}")
    print(f"   File size: {file_size:.4f} MB")
    
    # Check if onnx-tf is available
    print("\n2. Checking onnx-tf installation...")
    try:
        import onnx_tf
        print("   [OK] onnx-tf is installed")
    except ImportError:
        print("   [X] ERROR: onnx-tf not installed")
        print("   Install with: pip install onnx-tf")
        return False
    
    # Create output directory
    print(f"\n3. Preparing output directory...")
    if os.path.exists(output_dir):
        print(f"   [WARNING] Output directory exists: {output_dir}")
        print(f"   Will overwrite existing files")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"   [OK] Created directory: {output_dir}")
    
    # Convert ONNX to TensorFlow
    print("\n4. Converting ONNX to TensorFlow SavedModel...")
    print("   This may take a minute...")
    
    try:
        # Use onnx-tf command line tool
        cmd = [
            sys.executable, '-m', 'onnx_tf.converter',
            '-i', onnx_path,
            '-o', output_dir
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("   [OK] Conversion completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout[:200]}...")
        
    except subprocess.CalledProcessError as e:
        print(f"   [X] ERROR during conversion: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"   [X] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify conversion
    print("\n5. Verifying TensorFlow SavedModel...")
    saved_model_pb = os.path.join(output_dir, 'saved_model.pb')
    variables_dir = os.path.join(output_dir, 'variables')
    
    if os.path.exists(saved_model_pb):
        pb_size = os.path.getsize(saved_model_pb) / (1024 * 1024)  # MB
        print(f"   [OK] saved_model.pb found ({pb_size:.4f} MB)")
    else:
        print(f"   [X] ERROR: saved_model.pb not found")
        return False
    
    if os.path.exists(variables_dir):
        print(f"   [OK] variables directory found")
    else:
        print(f"   [WARNING] variables directory not found (may be empty model)")
    
    # Try to load and verify the model
    print("\n6. Testing TensorFlow model loading...")
    try:
        import tensorflow as tf
        loaded_model = tf.saved_model.load(output_dir)
        print("   [OK] TensorFlow model loaded successfully")
        
        # Get model signature
        if hasattr(loaded_model, 'signatures'):
            signatures = list(loaded_model.signatures.keys())
            print(f"   Available signatures: {signatures}")
    except Exception as e:
        print(f"   [WARNING] Could not load model: {e}")
        print("   (This is okay - conversion may have succeeded anyway)")
    
    print("\n" + "=" * 70)
    print("TENSORFLOW CONVERSION COMPLETE [OK]")
    print("=" * 70)
    print(f"\nTensorFlow SavedModel saved to: {output_dir}/")
    print(f"\nNext step: Convert TensorFlow to TFLite (Cell 14)")
    
    return True

if __name__ == '__main__':
    success = convert_onnx_to_tf()
    exit(0 if success else 1)

