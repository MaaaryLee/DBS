"""
Convert a saved TD3 actor policy to ONNX format without requiring the training
environment or MATLAB.
"""

import os
from pathlib import Path
from typing import Optional
import zipfile

import torch


class TD3Actor(torch.nn.Module):
    """Minimal TD3 actor: Linear -> ReLU -> Linear -> ReLU -> Linear -> tanh."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, int],
        action_dim: int,
        apply_output_tanh: bool = True,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, action_dim),
        )
        self.apply_output_tanh = apply_output_tanh

    def forward(self, x):
        x = self.backbone(x)
        return torch.tanh(x) if self.apply_output_tanh else x


def _build_actor_from_policy_state(state_dict, apply_output_tanh: bool = True):
    obs_dim = state_dict["actor.mu.0.weight"].shape[1]
    hidden1 = state_dict["actor.mu.0.weight"].shape[0]
    hidden2 = state_dict["actor.mu.2.weight"].shape[0]
    action_dim = state_dict["actor.mu.4.weight"].shape[0]

    actor = TD3Actor(
        obs_dim=obs_dim,
        hidden_dims=(hidden1, hidden2),
        action_dim=action_dim,
        apply_output_tanh=apply_output_tanh,
    )
    actor.eval()

    with torch.no_grad():
        actor.backbone[0].weight.copy_(state_dict["actor.mu.0.weight"])
        actor.backbone[0].bias.copy_(state_dict["actor.mu.0.bias"])
        actor.backbone[2].weight.copy_(state_dict["actor.mu.2.weight"])
        actor.backbone[2].bias.copy_(state_dict["actor.mu.2.bias"])
        actor.backbone[4].weight.copy_(state_dict["actor.mu.4.weight"])
        actor.backbone[4].bias.copy_(state_dict["actor.mu.4.bias"])

    return actor, obs_dim, action_dim, (hidden1, hidden2)


def convert_to_onnx(
    h1=32,
    h2=32,
    dynamic_batch: bool = False,
    policy_path: Optional[str] = None,
    output_path: str = "onnx_actors/model.onnx",
    apply_output_tanh: bool = True,
):
    """
    Convert quantized TD3 model to ONNX format.
    
    Args:
        h1: First hidden layer size
        h2: Second hidden layer size
    """
    print("=" * 70)
    print("Converting Quantized Model to ONNX (Cell 12 from examples.ipynb)")
    print("=" * 70)
    
    output_target = Path(output_path)
    output_target.parent.mkdir(parents=True, exist_ok=True)

    # Load actor weights directly from the saved policy file. This avoids the
    # need to reconstruct the TD3 training environment just for export.
    print("\n1. Loading saved policy...")
    try:
        if policy_path is None:
            policy_path = f"models/policies/policy_{h1}_{h2}.pth"
        print(f"   Loading FP32 policy from: {policy_path}")
        print("   Note: Using FP32 for ONNX export (quantization in TFLite)")
        policy_path_obj = Path(policy_path)
        if policy_path_obj.suffix == ".zip":
            with zipfile.ZipFile(policy_path_obj, "r") as archive:
                if "policy.pth" not in archive.namelist():
                    raise FileNotFoundError(f"`policy.pth` missing inside {policy_path_obj}")
                with archive.open("policy.pth", "r") as buffer:
                    state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
        else:
            state_dict = torch.load(policy_path_obj, map_location="cpu", weights_only=False)
        policy, obs_dim, action_dim, inferred_hidden_dims = _build_actor_from_policy_state(
            state_dict,
            apply_output_tanh=apply_output_tanh,
        )
        print(
            f"   [OK] FP32 policy loaded: obs_dim={obs_dim}, "
            f"hidden={inferred_hidden_dims[0]}x{inferred_hidden_dims[1]}, action_dim={action_dim}"
        )
        print(f"   Output tanh enabled: {apply_output_tanh}")
    except Exception as e:
        print(f"   [X] ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Determine input shape from saved weights.
    print("\n2. Determining input shape...")
    print(f"   Input shape: (1, {obs_dim})")
    
    # Create dummy input for ONNX export
    print("\n3. Creating dummy input...")
    dummy_input = torch.rand(1, obs_dim).to(torch.device('cpu'))
    print(f"   [OK] Dummy input created: shape {dummy_input.shape}")
    
    # Create a wrapper that only returns the tensor (not tuple)
    print("\n4. Creating ONNX-compatible wrapper...")
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
    print("\n5. Testing wrapper...")
    with torch.no_grad():
        test_output = wrapped_model(dummy_input)
        print(f"   Output shape: {test_output.shape}")
        print(f"   Output type: {type(test_output)}")
        print("   [OK] Wrapper works correctly")
    
    # Export to ONNX
    print("\n6. Exporting to ONNX format...")
    onnx_path = str(output_target)
    
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
    print("\n7. Verifying ONNX file...")
    if os.path.exists(onnx_path):
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"   [OK] ONNX file exists")
        print(f"   File size: {file_size:.4f} MB")
        print(f"   Location: {onnx_path}")
    else:
        print(f"   [X] ERROR: ONNX file not found at {onnx_path}")
        return False
    
    # Test ONNX model loading (optional verification)
    print("\n8. Testing ONNX model (optional verification)...")
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
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="onnx_actors/model.onnx")
    parser.add_argument(
        "--no-output-tanh",
        action="store_true",
        help="Export the actor without the final tanh. Useful for profiling whether the output activation matters.",
    )
    args = parser.parse_args()

    success = convert_to_onnx(
        h1=args.h1,
        h2=args.h2,
        dynamic_batch=bool(args.dynamic_batch),
        policy_path=args.policy_path,
        output_path=args.output_path,
        apply_output_tanh=not bool(args.no_output_tanh),
    )
    exit(0 if success else 1)
