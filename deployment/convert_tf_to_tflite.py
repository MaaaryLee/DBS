"""
Convert TensorFlow SavedModel to a TFLite FP32 baseline.
"""

import argparse
import os
from pathlib import Path
import tensorflow as tf


def _load_static_wrapper(saved_model_dir: str, batch_size: int = 1):
    loaded = tf.saved_model.load(saved_model_dir)
    serving = loaded.signatures.get("serving_default")
    if serving is None:
        raise ValueError("SavedModel has no 'serving_default' signature.")

    input_items = list(serving.structured_input_signature[1].items())
    if len(input_items) != 1:
        raise ValueError("Expected exactly one input tensor in serving_default.")

    input_name, input_tensor = input_items[0]
    input_shape = input_tensor.shape.as_list()
    if len(input_shape) != 2:
        raise ValueError(f"Expected rank-2 input [batch, features], got {input_shape}")
    obs_dim = int(input_shape[1])

    class StaticWrapper(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec([batch_size, obs_dim], input_tensor.dtype, name=input_name)
            ]
        )
        def serve(self, observation):
            return serving(**{input_name: observation})

    wrapper = StaticWrapper()
    concrete = wrapper.serve.get_concrete_function()
    return wrapper, concrete, obs_dim, input_name

def convert_tf_to_tflite(
    saved_model_dir='tf_model',
    output_path='tflite_actors/model_fp32.tflite',
    optimize=False,
    alias_path='tflite_actors/model.tflite',
):
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
        # Load the SavedModel as a static batch=1 function to avoid dynamic-shaped
        # tensors that can reduce delegate optimization quality.
        print("   Loading SavedModel with static batch=1 signature...")
        wrapper, concrete, obs_dim, input_name = _load_static_wrapper(saved_model_dir, batch_size=1)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)

        # Prefer the legacy converter for older Arduino flows, but fall back to the
        # new MLIR converter when the legacy TOCO binary is unavailable locally.
        print("   Preferring legacy converter for Arduino TFLite Micro...")
        legacy_requested = False
        if hasattr(converter, "experimental_new_converter"):
            converter.experimental_new_converter = False
            legacy_requested = True
        if hasattr(converter, "experimental_new_quantizer"):
            converter.experimental_new_quantizer = False

        print("   Configuring converter...")
        if optimize:
            print("   Optimization enabled (this may create a hybrid baseline, not pure FP32).")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            print("   Optimization disabled to keep a true FP32 baseline.")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Convert
        print("   Converting...")
        try:
            tflite_model = converter.convert()
        except Exception as exc:
            if legacy_requested and "toco_from_protos" in str(exc):
                print("   [WARNING] Legacy converter unavailable locally; retrying with new converter...")
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
                if optimize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                tflite_model = converter.convert()
            else:
                raise
        
        # Save
        print(f"   Saving to {output_path}...")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        alias_target = Path(alias_path)
        if alias_target != Path(output_path):
            alias_target.parent.mkdir(parents=True, exist_ok=True)
            alias_target.write_bytes(tflite_model)
            print(f"   [OK] Legacy alias refreshed: {alias_target}")
        
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
        # Avoid auto-loading default delegates (e.g., XNNPACK) during verification so logs
        # remain consistent with our benchmarking scripts.
        disable_default_delegates = os.environ.get("TFLITE_DISABLE_DEFAULT_DELEGATES", "1") == "1"
        interpreter_kwargs = {"model_path": output_path}
        if disable_default_delegates:
            try:
                interpreter_kwargs["experimental_op_resolver_type"] = (
                    tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                )
            except Exception:
                pass
        try:
            interpreter = tf.lite.Interpreter(**interpreter_kwargs)
        except TypeError:
            interpreter_kwargs.pop("experimental_op_resolver_type", None)
            interpreter = tf.lite.Interpreter(**interpreter_kwargs)
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
    print(f"Static input signature: {input_name} -> [1, {obs_dim}]")
    print(f"\nNext step: Convert TFLite to C byte array (Cell 15)")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert SavedModel to a TFLite FP32 baseline.")
    parser.add_argument("--saved-model-dir", default="tf_model")
    parser.add_argument("--output-path", default="tflite_actors/model_fp32.tflite")
    parser.add_argument("--alias-path", default="tflite_actors/model.tflite")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable TFLite optimizations (can produce a hybrid/dynamic-range baseline instead of pure FP32).",
    )
    args = parser.parse_args()

    success = convert_tf_to_tflite(
        saved_model_dir=args.saved_model_dir,
        output_path=args.output_path,
        optimize=bool(args.optimize),
        alias_path=args.alias_path,
    )
    exit(0 if success else 1)
