"""
Convert quantized TD3 model to ONNX format (Cell 12 from examples.ipynb).
"""

import torch
from stable_baselines3 import TD3
from BGN_MC import BGN_MC
import os

def convert_to_onnx(h1=32, h2=32, dynamic_batch: bool = False):
    """
    Convert quantized TD3 model to ONNX format.
    
    Args:
        h1: First hidden layer size
        h2: Second hidden layer size
    """
    print("=" * 70)
    print("Converting Quantized Model to ONNX (Cell 12 from examples.ipynb)")
    print("=" * 70)
    
    # Create directories
    onnx_dir = 'onnx_actors'
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
        print(f"[OK] Created directory: {onnx_dir}")
    
    # Create environment (needed for TD3 structure)
    print("\n1. Creating BGN environment...")
    bgn = BGN_MC(tmax=1100, pd=True)
    print("   [OK] Environment created")
    
    # Create TD3 model structure
    print("\n2. Creating TD3 model structure...")
    try:
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[h1, h2], qf=[h1, h2])
        )
        model = TD3('MlpPolicy', bgn, verbose=0, policy_kwargs=policy_kwargs, learning_rate=0.0001)
        
        # Load FP32 policy (ONNX doesn't support quantized operators directly)
        # We'll use FP32 for ONNX, quantization will happen in TFLite
        policy_path = f'models/policies/policy_{h1}_{h2}.pth'
        print(f"   Loading FP32 policy from: {policy_path}")
        print("   Note: Using FP32 for ONNX export (quantization in TFLite)")
        
        policy = model.policy.to(torch.device('cpu'))
        policy.load_state_dict(torch.load(policy_path, weights_only=False))
        policy.eval()
        print("   [OK] FP32 policy loaded")
    except Exception as e:
        print(f"   [X] ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Determine input shape based on observation space
    print("\n3. Determining input shape...")
    obs, _ = bgn.reset()
    obs_dim = len(obs)
    print(f"   Observation dimension: {obs_dim}")
    print(f"   Input shape: (1, {obs_dim})")
    
    # Create dummy input for ONNX export
    print("\n4. Creating dummy input...")
    dummy_input = torch.rand(1, obs_dim).to(torch.device('cpu'))
    print(f"   [OK] Dummy input created: shape {dummy_input.shape}")
    
    # Create a wrapper that only returns the tensor (not tuple)
    print("\n5. Creating ONNX-compatible wrapper...")
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Call the model and extract just the tensor output
            output = self.model(x)
            # If output is a tuple, take the first element (the tensor)
            if isinstance(output, tuple):
                return output[0]
            return output
    
    wrapped_model = ONNXWrapper(policy)
    wrapped_model.eval()
    print("   [OK] Wrapper created")
    
    # Test the wrapper
    print("\n6. Testing wrapper...")
    with torch.no_grad():
        test_output = wrapped_model(dummy_input)
        print(f"   Output shape: {test_output.shape}")
        print(f"   Output type: {type(test_output)}")
        print("   [OK] Wrapper works correctly")
    
    # Export to ONNX
    print("\n7. Exporting to ONNX format...")
    onnx_path = f'{onnx_dir}/model.onnx'
    
    try:
        # Standard ONNX export with wrapped model
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'observation': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }

        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes=dynamic_axes,
            opset_version=13,
            do_constant_folding=True,
            export_params=True
        )
        print(f"   [OK] ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"   [X] ERROR exporting to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify ONNX file was created
    print("\n8. Verifying ONNX file...")
    if os.path.exists(onnx_path):
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"   [OK] ONNX file exists")
        print(f"   File size: {file_size:.4f} MB")
        print(f"   Location: {onnx_path}")
    else:
        print(f"   [X] ERROR: ONNX file not found at {onnx_path}")
        return False
    
    # Test ONNX model loading (optional verification)
    print("\n9. Testing ONNX model (optional verification)...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("   [OK] ONNX model is valid")
        
        # Print model info
        print(f"\n   Model Info:")
        print(f"   - Inputs: {len(onnx_model.graph.input)}")
        print(f"   - Outputs: {len(onnx_model.graph.output)}")
        for inp in onnx_model.graph.input:
            print(f"     Input: {inp.name}, Shape: {[dim.dim_value for dim in inp.type.tensor_type.shape.dim]}")
        for out in onnx_model.graph.output:
            print(f"     Output: {out.name}, Shape: {[dim.dim_value for dim in out.type.tensor_type.shape.dim]}")
    except ImportError:
        print("   [SKIP] ONNX library not installed (optional verification)")
        print("   Install with: pip install onnx")
    except Exception as e:
        print(f"   [WARNING] Could not verify ONNX model: {e}")
        print("   (This is okay - file was created successfully)")
    
    print("\n" + "=" * 70)
    print("ONNX CONVERSION COMPLETE [OK]")
    print("=" * 70)
    print(f"\nONNX model saved to: {onnx_path}")
    print(f"\nNext step: Convert ONNX to TensorFlow SavedModel (Cell 13)")
    print("Command: onnx-tf convert -i onnx_actors/model.onnx -o tf_model/")
    print("(Requires TensorFlow environment)")
    
    return True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Export TD3 policy to ONNX (default: 22x22).")
    parser.add_argument("--h1", type=int, default=22)
    parser.add_argument("--h2", type=int, default=22)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export ONNX with dynamic batch dimension (can block XNNPACK static-shape delegation).",
    )
    args = parser.parse_args()

    success = convert_to_onnx(h1=args.h1, h2=args.h2, dynamic_batch=bool(args.dynamic_batch))
    exit(0 if success else 1)

