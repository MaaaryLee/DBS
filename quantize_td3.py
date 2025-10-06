"""
Quantize TD3 actor using weights-only INT8 Post-Training Quantization (PTQ).
Measures fidelity, performance, and model size.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from stable_baselines3 import TD3
from BGN_MC_no_matlab import BGN_MC_NoMatlab
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def load_models(model_path='models/TD3_64_64/1500.zip'):
    """Load FP32 and create quantized models."""
    print("Loading FP32 model...")
    env = BGN_MC_NoMatlab(tmax=1100, pd=True)
    model = TD3.load(model_path, env=env)
    
    # Extract policy (actor network)
    policy = model.policy.actor.to(torch.device('cpu'))
    policy.eval()
    
    print("Creating quantized model...")
    # Quantize weights-only (INT8), keep activations FP32
    quantized_policy = quantize_dynamic(policy, dtype=torch.qint8)
    
    return policy, quantized_policy, env

def measure_model_size(model, model_name):
    """Measure model size in MB."""
    # Save model to temporary file and measure size
    temp_path = f'temp_{model_name}.pth'
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_mb

def measure_fidelity(fp32_policy, quantized_policy, states_path='states_eval.npy'):
    """Measure policy fidelity (MSE between FP32 and INT8 actions)."""
    print("Measuring policy fidelity...")
    
    # Load calibration states
    states = np.load(states_path)
    states_tensor = torch.from_numpy(states).float()
    
    # Get actions from both models
    with torch.no_grad():
        fp32_actions = fp32_policy(states_tensor).numpy()
        quantized_actions = quantized_policy(states_tensor).numpy()
    
    # Calculate MSE
    mse = mean_squared_error(fp32_actions, quantized_actions)
    
    print(f"Policy fidelity (MSE): {mse:.6f}")
    print(f"Max action difference: {np.max(np.abs(fp32_actions - quantized_actions)):.6f}")
    print(f"Mean action difference: {np.mean(np.abs(fp32_actions - quantized_actions)):.6f}")
    
    return mse, fp32_actions, quantized_actions

def evaluate_performance(policy, env, num_episodes=20, model_name="Model"):
    """Evaluate model performance over multiple episodes."""
    print(f"Evaluating {model_name} performance over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = 0
        episode_reward = 0
        episode_length = 0
        
        while terminated != 1:
            # Convert observation to tensor format expected by policy
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            with torch.no_grad():
                action = policy(obs_tensor).numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.4f}")
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"{model_name} Results:")
    print(f"  Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"  Average episode length: {avg_length:.1f}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'episode_rewards': episode_rewards
    }

def measure_power_usage(policy, states_path='states_eval.npy', num_runs=100):
    """Measure average power usage (simulated via inference time)."""
    print("Measuring power usage (inference time)...")
    
    states = np.load(states_path)
    states_tensor = torch.from_numpy(states[:1000]).float()  # Use subset for timing
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = policy(states_tensor)
    
    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = policy(states_tensor)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    
    return avg_time, std_time

def create_comparison_plots(results):
    """Create comparison plots for accuracy vs bit-width and size vs bit-width."""
    print("Creating comparison plots...")
    
    # Prepare data
    bit_widths = [32, 8]  # FP32 and INT8
    accuracies = [1.0, 1.0 - results['mse']]  # Normalized accuracy (1 - MSE)
    model_sizes = [results['fp32_size'], results['quantized_size']]
    avg_rewards = [results['fp32_performance']['avg_reward'], 
                   results['quantized_performance']['avg_reward']]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Accuracy vs Bit-width
    ax1.plot(bit_widths, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Bit-width')
    ax1.set_ylabel('Policy Fidelity (1 - MSE)')
    ax1.set_title('Policy Fidelity vs Bit-width')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(bit_widths)
    
    # Plot 2: Model Size vs Bit-width
    ax2.plot(bit_widths, model_sizes, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Bit-width')
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size vs Bit-width')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(bit_widths)
    
    # Plot 3: Performance vs Bit-width
    ax3.plot(bit_widths, avg_rewards, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Bit-width')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Performance vs Bit-width')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(bit_widths)
    
    # Plot 4: Size reduction
    size_reduction = (results['fp32_size'] - results['quantized_size']) / results['fp32_size'] * 100
    ax4.bar(['FP32', 'INT8'], model_sizes, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Model Size (MB)')
    ax4.set_title(f'Model Size Comparison\n({size_reduction:.1f}% reduction)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main quantization and evaluation pipeline."""
    print("=== TD3 Actor Quantization Pipeline ===")
    
    # Load models
    fp32_policy, quantized_policy, env = load_models()
    
    # Measure model sizes
    print("\n=== Model Size Analysis ===")
    fp32_size = measure_model_size(fp32_policy, 'fp32')
    quantized_size = measure_model_size(quantized_policy, 'quantized')
    
    print(f"FP32 model size: {fp32_size:.2f} MB")
    print(f"INT8 model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {((fp32_size - quantized_size) / fp32_size * 100):.1f}%")
    
    # Measure fidelity
    print("\n=== Policy Fidelity Analysis ===")
    mse, fp32_actions, quantized_actions = measure_fidelity(fp32_policy, quantized_policy)
    
    # Evaluate performance
    print("\n=== Performance Evaluation ===")
    fp32_performance = evaluate_performance(fp32_policy, env, num_episodes=20, model_name="FP32")
    quantized_performance = evaluate_performance(quantized_policy, env, num_episodes=20, model_name="INT8")
    
    # Measure power usage
    print("\n=== Power Usage Analysis ===")
    fp32_time, fp32_std = measure_power_usage(fp32_policy, model_name="FP32")
    quantized_time, quantized_std = measure_power_usage(quantized_policy, model_name="INT8")
    
    # Compile results
    results = {
        'mse': mse,
        'fp32_size': fp32_size,
        'quantized_size': quantized_size,
        'fp32_performance': fp32_performance,
        'quantized_performance': quantized_performance,
        'fp32_time': fp32_time,
        'quantized_time': quantized_time,
        'fp32_actions': fp32_actions,
        'quantized_actions': quantized_actions
    }
    
    # Create plots
    print("\n=== Creating Visualization ===")
    create_comparison_plots(results)
    
    # Print summary
    print("\n=== QUANTIZATION SUMMARY ===")
    print(f"Policy Fidelity (MSE): {mse:.6f}")
    print(f"Model Size Reduction: {((fp32_size - quantized_size) / fp32_size * 100):.1f}%")
    print(f"Performance Drop: {fp32_performance['avg_reward'] - quantized_performance['avg_reward']:.4f}")
    print(f"Speed Improvement: {fp32_time / quantized_time:.2f}x")
    
    # Save quantized model
    torch.save(quantized_policy.state_dict(), 'models/policies/qpolicy_64_64.pth')
    torch.save(fp32_policy.state_dict(), 'models/policies/policy_64_64.pth')
    print("\nQuantized model saved to models/policies/qpolicy_64_64.pth")
    
    return results

if __name__ == "__main__":
    results = main()
