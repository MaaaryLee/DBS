"""
Quantize TD3 model following examples.ipynb Cell 10.
Converts FP32 model to INT8 using PyTorch's quantize_dynamic.
"""

import torch
from torch.ao.quantization import quantize_dynamic
from stable_baselines3 import TD3
from BGN_MC import BGN_MC
import os

def quantize_td3_model(h1=32, h2=32, model_timesteps=2500):
    """
    Quantize a trained TD3 model following examples.ipynb Cell 10.
    
    Args:
        h1: First hidden layer size
        h2: Second hidden layer size  
        model_timesteps: Timesteps of the model to load (e.g., 2500)
    """
    print("=" * 70)
    print("TD3 Model Quantization (Cell 10 from examples.ipynb)")
    print("=" * 70)
    
    # Define paths
    model_path = f'models/TD3_{h1}_{h2}/{model_timesteps}.zip'
    policy_dir = 'models/policies'
    
    # Create policies directory if it doesn't exist
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)
        print(f"[OK] Created directory: {policy_dir}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[X] ERROR: Model not found at {model_path}")
        print(f"Available models:")
        model_dir = f'models/TD3_{h1}_{h2}'
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                print(f"  - {f}")
        return False
    
    print(f"\n1. Loading trained model from {model_path}...")
    try:
        # Create environment (needed for TD3.load)
        env = BGN_MC(tmax=1100, pd=True)
        model = TD3.load(model_path, env=env)
        print("   [OK] Model loaded successfully")
    except Exception as e:
        print(f"   [X] ERROR loading model: {e}")
        return False
    
    print("\n2. Extracting policy (actor network)...")
    try:
        # Extract policy and move to CPU
        policy = model.policy.to(torch.device('cpu'))
        policy.eval()  # Set to evaluation mode
        print("   [OK] Policy extracted and set to eval mode")
    except Exception as e:
        print(f"   [X] ERROR extracting policy: {e}")
        return False
    
    print("\n3. Performing dynamic quantization (FP32 -> INT8)...")
    try:
        # Perform dynamic quantization
        # This quantizes weights to INT8 but keeps activations in FP32
        qpolicy = quantize_dynamic(policy, dtype=torch.qint8)
        print("   [OK] Quantization completed")
    except Exception as e:
        print(f"   [X] ERROR during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n4. Saving quantized and original policies...")
    try:
        # Save quantized policy
        qpolicy_path = f'{policy_dir}/qpolicy_{h1}_{h2}.pth'
        torch.save(qpolicy.state_dict(), qpolicy_path)
        print(f"   [OK] Quantized policy saved to: {qpolicy_path}")
        
        # Save original FP32 policy
        policy_path = f'{policy_dir}/policy_{h1}_{h2}.pth'
        torch.save(policy.state_dict(), policy_path)
        print(f"   [OK] Original policy saved to: {policy_path}")
    except Exception as e:
        print(f"   [X] ERROR saving policies: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5. Calculating model sizes...")
    try:
        # Calculate file sizes
        qpolicy_size = os.path.getsize(qpolicy_path) / (1024 * 1024)  # MB
        policy_size = os.path.getsize(policy_path) / (1024 * 1024)  # MB
        reduction = (1 - qpolicy_size / policy_size) * 100
        
        print(f"   Original (FP32) size: {policy_size:.4f} MB")
        print(f"   Quantized (INT8) size: {qpolicy_size:.4f} MB")
        print(f"   Size reduction: {reduction:.2f}%")
    except Exception as e:
        print(f"   [WARNING] Could not calculate sizes: {e}")
    
    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE [OK]")
    print("=" * 70)
    print(f"\nQuantized model ready for inference testing!")
    print(f"Next step: Test quantized model (Cell 11 from examples.ipynb)")
    return True

if __name__ == '__main__':
    # Use the same architecture as training.py
    h1, h2 = 32, 32
    model_timesteps = 2500  # Use the final trained model
    
    success = quantize_td3_model(h1=h1, h2=h2, model_timesteps=model_timesteps)
    exit(0 if success else 1)

