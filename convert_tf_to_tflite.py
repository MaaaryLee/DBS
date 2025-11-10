"""
Convert TensorFlow SavedModel to TFLite format (Cell 14 from examples.ipynb).
"""

import os
import tensorflow as tf

def convert_tf_to_tflite(saved_model_dir='tf_model', output_path='tflite_actors/model.tflite'):
    """
    Convert TensorFlow SavedModel to TFLite format.
    
    Args:
        saved_model_dir: Directory containing TensorFlow SavedModel
        output_path: Path to save TFLite model
    """
    print("=" * 70)
    print("Converting TensorFlow to TFLite (Cell 14 from examples.ipynb)")
    print("=" * 70)
    
    # Check if SavedModel exists
    print(f"\n1. Checking TensorFlow SavedModel...")
    saved_model_pb = os.path.join(saved_model_dir, 'saved_model.pb')
    if not os.path.exists(saved_model_pb):
        print(f"   [X] ERROR: SavedModel not found at {saved_model_dir}")
        return False
    
    pb_size = os.path.getsize(saved_model_pb) / (1024 * 1024)  # MB
    print(f"   [OK] SavedModel found: {saved_model_pb}")
    print(f"   File size: {pb_size:.4f} MB")
    
    # Create output directory
    print(f"\n2. Preparing output directory...")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"   [OK] Created directory: {output_dir}")
    elif output_dir:
        print(f"   [OK] Output directory exists: {output_dir}")
    
    # Convert to TFLite
    print("\n3. Converting to TFLite format...")
    try:
        # Load the SavedModel
        print("   Loading SavedModel...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        # Set optimizations (quantization will happen here)
        print("   Setting optimizations...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        print("   Converting...")
        tflite_model = converter.convert()
        
        # Save
        print(f"   Saving to {output_path}...")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"   [OK] TFLite model saved successfully")
        
    except Exception as e:
        print(f"   [X] ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify TFLite file
    print("\n4. Verifying TFLite file...")
    if os.path.exists(output_path):
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"   [OK] TFLite file exists")
        print(f"   File size: {tflite_size:.4f} MB")
        print(f"   Location: {output_path}")
    else:
        print(f"   [X] ERROR: TFLite file not found")
        return False
    
    # Test loading TFLite model (optional)
    print("\n5. Testing TFLite model (optional verification)...")
    try:
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("   [OK] TFLite model loads successfully")
        print(f"\n   Model Info:")
        print(f"   - Inputs: {len(input_details)}")
        for i, inp in enumerate(input_details):
            print(f"     Input {i}: {inp['name']}, Shape: {inp['shape']}, Type: {inp['dtype']}")
        print(f"   - Outputs: {len(output_details)}")
        for i, out in enumerate(output_details):
            print(f"     Output {i}: {out['name']}, Shape: {out['shape']}, Type: {out['dtype']}")
    except Exception as e:
        print(f"   [WARNING] Could not verify TFLite model: {e}")
        print("   (This is okay - file was created successfully)")
    
    print("\n" + "=" * 70)
    print("TFLITE CONVERSION COMPLETE [OK]")
    print("=" * 70)
    print(f"\nTFLite model saved to: {output_path}")
    print(f"\nNext step: Convert TFLite to C byte array (Cell 15)")
    
    return True

if __name__ == '__main__':
    success = convert_tf_to_tflite()
    exit(0 if success else 1)

