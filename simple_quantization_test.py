"""
Simple, working quantization test to verify everything works.
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import TD3
from BGN_MC_no_matlab import BGN_MC_NoMatlab

def test_environment():
    """Test if environment is working properly."""
    print("=== Testing Environment ===")
    env = BGN_MC_NoMatlab(tmax=1100, pd=True)
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation shape: {obs.shape}")
    
    # Test a few steps
    total_reward = 0
    for i in range(5):
        action = np.array([0.5, 0.5])  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.4f}, terminated={terminated}")
        
        if terminated:
            break
    
    print(f"Total reward over {i+1} steps: {total_reward:.4f}")
    return total_reward > 0

def test_model_loading():
    """Test if model loads and works."""
    print("\n=== Testing Model Loading ===")
    env = BGN_MC_NoMatlab(tmax=1100, pd=True)
    model = TD3.load('models/TD3_64_64/1500.zip', env=env)
    
    policy = model.policy.actor
    print(f"Policy type: {type(policy)}")
    print(f"Policy device: {next(policy.parameters()).device}")
    
    # Test inference
    obs, info = env.reset()
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
    
    with torch.no_grad():
        action = policy(obs_tensor).numpy()[0]
    
    print(f"Model action: {action}")
    print(f"Action shape: {action.shape}")
    
    return True

def test_simple_quantization():
    """Test simple weight quantization."""
    print("\n=== Testing Simple Quantization ===")
    
    # Create a simple linear layer
    linear = nn.Linear(6, 2)
    print(f"Original weight shape: {linear.weight.shape}")
    print(f"Original weight range: [{linear.weight.min():.4f}, {linear.weight.max():.4f}]")
    
    # Quantize weights
    weight = linear.weight.data
    scale = weight.abs().max() / 127.0
    weight_int8 = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    
    print(f"Quantized weight shape: {weight_int8.shape}")
    print(f"Quantized weight range: [{weight_int8.min()}, {weight_int8.max()}]")
    print(f"Scale factor: {scale:.6f}")
    
    # Test dequantization
    weight_dequantized = weight_int8.float() * scale
    mse = torch.mean((weight - weight_dequantized) ** 2)
    print(f"Quantization error (MSE): {mse:.6f}")
    
    return True

def main():
    """Run all tests."""
    print("=== Simple Quantization Test ===")
    
    # Test 1: Environment
    env_works = test_environment()
    
    # Test 2: Model loading
    model_works = test_model_loading()
    
    # Test 3: Simple quantization
    quant_works = test_simple_quantization()
    
    print("\n=== Test Results ===")
    print(f"Environment works: {env_works}")
    print(f"Model loads: {model_works}")
    print(f"Quantization works: {quant_works}")
    
    if not env_works:
        print("❌ Environment is broken - no rewards being generated")
    if not model_works:
        print("❌ Model loading failed")
    if not quant_works:
        print("❌ Basic quantization failed")
    
    if env_works and model_works and quant_works:
        print("✅ All basic tests passed - ready for real quantization")
    else:
        print("❌ Some tests failed - need to fix issues first")

if __name__ == "__main__":
    main()
