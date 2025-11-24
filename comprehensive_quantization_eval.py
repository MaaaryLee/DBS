"""
Comprehensive quantization evaluation script.
Compares FP32 vs INT8 models on accuracy, size, performance, and power metrics.
"""

import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from stable_baselines3 import TD3
from BGN_MC import BGN_MC
from torch.ao.quantization import quantize_dynamic
import scipy.io


def load_models(h1=32, h2=32, model_timesteps=2500):
    """
    Load FP32 and INT8 models.
    
    Returns:
        Tuple of (fp32_policy, int8_policy, env)
    """
    model_path = f'models/TD3_{h1}_{h2}/{model_timesteps}.zip'
    qpolicy_path = f'models/policies/qpolicy_{h1}_{h2}.pth'
    policy_path = f'models/policies/policy_{h1}_{h2}.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(qpolicy_path):
        raise FileNotFoundError(f"Quantized policy not found: {qpolicy_path}")
    
    # Load FP32 model
    env = BGN_MC(tmax=1100, pd=True)
    model = TD3.load(model_path, env=env)
    fp32_policy = model.policy.to(torch.device('cpu'))
    fp32_policy.eval()
    
    # Load INT8 policy
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[h1, h2], qf=[h1, h2])
    )
    dummy_model = TD3('MlpPolicy', env, verbose=0, policy_kwargs=policy_kwargs, learning_rate=0.0001)
    int8_policy = quantize_dynamic(dummy_model.policy.to(torch.device('cpu')), dtype=torch.qint8)
    int8_policy.load_state_dict(torch.load(qpolicy_path, weights_only=False))
    int8_policy.eval()
    
    return fp32_policy, int8_policy, env


def measure_model_size(h1=32, h2=32):
    """Measure model file sizes."""
    policy_path = f'models/policies/policy_{h1}_{h2}.pth'
    qpolicy_path = f'models/policies/qpolicy_{h1}_{h2}.pth'
    
    fp32_size = os.path.getsize(policy_path) / (1024 * 1024)  # MB
    int8_size = os.path.getsize(qpolicy_path) / (1024 * 1024)  # MB
    reduction = (1 - int8_size / fp32_size) * 100
    
    return {
        'fp32_size_mb': fp32_size,
        'int8_size_mb': int8_size,
        'reduction_percent': reduction
    }


def measure_fidelity(fp32_policy, int8_policy, states_path='states_eval.npy'):
    """
    Measure policy fidelity (MSE between FP32 and INT8 actions).
    
    Returns:
        Dictionary with MSE and action differences
    """
    if not os.path.exists(states_path):
        print(f"[WARNING] States file not found: {states_path}")
        print("  Collecting calibration states...")
        states = collect_calibration_states()
        np.save(states_path, states)
    else:
        states = np.load(states_path)
    
    states_tensor = torch.from_numpy(states).float()
    
    # Get actions from both models
    with torch.no_grad():
        fp32_actions = fp32_policy(states_tensor).numpy()
        int8_actions = int8_policy(states_tensor).numpy()
    
    # Calculate metrics
    mse = mean_squared_error(fp32_actions, int8_actions)
    max_diff = np.max(np.abs(fp32_actions - int8_actions))
    mean_diff = np.mean(np.abs(fp32_actions - int8_actions))
    std_diff = np.std(np.abs(fp32_actions - int8_actions))
    
    return {
        'mse': float(mse),
        'max_action_diff': float(max_diff),
        'mean_action_diff': float(mean_diff),
        'std_action_diff': float(std_diff),
        'fp32_actions': fp32_actions.tolist(),
        'int8_actions': int8_actions.tolist()
    }


def collect_calibration_states(num_states=1000):
    """Collect calibration states from the environment."""
    env = BGN_MC(tmax=1100, pd=True)
    states = []
    
    for _ in range(num_states):
        obs, _ = env.reset()
        states.append(obs)
        # Take a few random steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    
    return np.array(states)


def measure_inference_time(policy, states_path='states_eval.npy', num_runs=100):
    """
    Measure inference time.
    
    Returns:
        Dictionary with timing statistics
    """
    if not os.path.exists(states_path):
        states = collect_calibration_states()
        np.save(states_path, states)
    else:
        states = np.load(states_path)
    
    states_tensor = torch.from_numpy(states[:100]).float()  # Use subset for timing
    
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
    
    arr = np.array(times)
    return {
        'mean_ms': float(arr.mean() * 1000),
        'std_ms': float(arr.std(ddof=0) * 1000),
        'min_ms': float(arr.min() * 1000),
        'max_ms': float(arr.max() * 1000),
        'p50_ms': float(np.percentile(arr, 50) * 1000),
        'p90_ms': float(np.percentile(arr, 90) * 1000)
    }


def evaluate_performance(policy, env, num_episodes=5, model_name="Model"):
    """
    Evaluate model performance over multiple episodes.
    
    Returns:
        Dictionary with performance metrics
    """
    episode_rewards = []
    episode_lengths = []
    sgis_sums = []
    Pbs = []
    freqs = []
    amps = []
    
    for episode in range(num_episodes):
        try:
            observation = env.reset()[0]
        except Exception as e:
            print(f"  [WARNING] Reset failed, recreating environment: {e}")
            env = BGN_MC(tmax=1100, pd=True)
            observation = env.reset()[0]
        
        terminated = False
        step_count = 0
        total_reward = 0
        
        while not terminated:
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(torch.device('cpu'))
            
            with torch.no_grad():
                action = policy(obs_tensor).numpy()[0]
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Collect data for metrics
            try:
                mat_data = scipy.io.loadmat('bgn_vars.mat')
                sgis = mat_data['sgis']
                vgi = mat_data['vgi']
                
                sgis_sums.append(np.sum(np.mean(np.abs(np.fft.fft(sgis)), axis=0)[1:20]))
                
                # Denormalize actions
                freqs.append(185 * ((action[0] + 1) / 2))
                amps.append(5000 * ((action[1] + 1) / 2))
            except Exception as e:
                pass  # Skip if MATLAB data not available
            
            step_count += 1
            if step_count >= env.tmax // 100:
                terminated = True
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # Calculate P-beta for the episode
        try:
            mat_data = scipy.io.loadmat('bgn_vars.mat')
            vgi_full = mat_data['vgi']
            current_i = mat_data['i'].flatten()[0]
            
            sim_time_ms = env.tmax
            slice_start = max(0, current_i - sim_time_ms)
            
            if vgi_full.shape[1] > slice_start:
                vgi_episode = vgi_full[:, slice_start:current_i:2]
                Pb = np.sum(np.average(np.abs(np.fft.fft(vgi_episode)) / 0.1, axis=0)[12:31])
                Pbs.append(Pb)
            else:
                Pbs.append(0)
        except Exception:
            Pbs.append(0)
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_sgi_intensity': float(np.mean(sgis_sums)) if sgis_sums else None,
        'mean_p_beta': float(np.mean(Pbs)) if Pbs else None,
        'mean_frequency': float(np.mean(freqs)) if freqs else None,
        'mean_amplitude': float(np.mean(amps)) if amps else None
    }


def create_comparison_plots(results, output_dir='quantization_eval_plots'):
    """Create comparison plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Model size comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [results['size']['fp32_size_mb'], results['size']['int8_size_mb']]
    labels = ['FP32', 'INT8']
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(labels, sizes, color=colors)
    ax.set_ylabel('Model Size (MB)', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.4f} MB',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_size_comparison.png', dpi=150)
    plt.show()  # Display the plot
    plt.close()
    
    # 2. Inference time comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    fp32_time = results['inference_time']['fp32']['mean_ms']
    int8_time = results['inference_time']['int8']['mean_ms']
    times = [fp32_time, int8_time]
    bars = ax.bar(labels, times, color=colors)
    ax.set_ylabel('Mean Inference Time (ms)', fontsize=12)
    ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f} ms',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/inference_time_comparison.png', dpi=150)
    plt.show()  # Display the plot
    plt.close()
    
    # 3. Action difference histogram
    if 'fidelity' in results and results['fidelity']['fp32_actions']:
        fig, ax = plt.subplots(figsize=(10, 6))
        fp32_actions = np.array(results['fidelity']['fp32_actions'])
        int8_actions = np.array(results['fidelity']['int8_actions'])
        differences = np.abs(fp32_actions - int8_actions).flatten()
        
        ax.hist(differences, bins=50, alpha=0.7, color='#9b59b6', edgecolor='black')
        ax.set_xlabel('Absolute Action Difference', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Action Difference Distribution (FP32 vs INT8)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axvline(np.mean(differences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(differences):.6f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/action_difference_histogram.png', dpi=150)
        plt.show()  # Display the plot
        plt.close()
    
    print(f"[OK] Plots saved to {output_dir}/")


def main():
    print("=" * 70)
    print("Comprehensive Quantization Evaluation")
    print("=" * 70)
    
    h1, h2 = 32, 32
    model_timesteps = 2500
    num_episodes = 5
    
    # Load models
    print("\n1. Loading models...")
    try:
        fp32_policy, int8_policy, env = load_models(h1=h1, h2=h2, model_timesteps=model_timesteps)
        print("   [OK] Models loaded successfully")
    except Exception as e:
        print(f"   [X] ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Measure model sizes
    print("\n2. Measuring model sizes...")
    size_results = measure_model_size(h1=h1, h2=h2)
    print(f"   FP32 size: {size_results['fp32_size_mb']:.4f} MB")
    print(f"   INT8 size: {size_results['int8_size_mb']:.4f} MB")
    print(f"   Size reduction: {size_results['reduction_percent']:.2f}%")
    
    # Measure fidelity
    print("\n3. Measuring policy fidelity...")
    fidelity_results = measure_fidelity(fp32_policy, int8_policy)
    print(f"   MSE: {fidelity_results['mse']:.6f}")
    print(f"   Max action difference: {fidelity_results['max_action_diff']:.6f}")
    print(f"   Mean action difference: {fidelity_results['mean_action_diff']:.6f}")
    
    # Measure inference time
    print("\n4. Measuring inference time...")
    fp32_time = measure_inference_time(fp32_policy)
    print(f"   FP32 mean: {fp32_time['mean_ms']:.4f} ms ± {fp32_time['std_ms']:.4f} ms")
    int8_time = measure_inference_time(int8_policy)
    print(f"   INT8 mean: {int8_time['mean_ms']:.4f} ms ± {int8_time['std_ms']:.4f} ms")
    speedup = fp32_time['mean_ms'] / int8_time['mean_ms'] if int8_time['mean_ms'] > 0 else 0
    print(f"   Speedup: {speedup:.2f}x")
    
    # Evaluate performance
    print(f"\n5. Evaluating performance ({num_episodes} episodes)...")
    print("   FP32 model...")
    fp32_perf = evaluate_performance(fp32_policy, env, num_episodes=num_episodes, model_name="FP32")
    print("   INT8 model...")
    int8_perf = evaluate_performance(int8_policy, env, num_episodes=num_episodes, model_name="INT8")
    
    print("\n   FP32 Performance:")
    if fp32_perf['mean_sgi_intensity']:
        print(f"     SGi Intensity: {fp32_perf['mean_sgi_intensity']:.2f}")
    if fp32_perf['mean_p_beta']:
        print(f"     P-beta: {fp32_perf['mean_p_beta']:.2f}")
    if fp32_perf['mean_frequency']:
        print(f"     Mean Frequency: {fp32_perf['mean_frequency']:.2f} Hz")
    if fp32_perf['mean_amplitude']:
        print(f"     Mean Amplitude: {fp32_perf['mean_amplitude']:.2f} mA")
    
    print("\n   INT8 Performance:")
    if int8_perf['mean_sgi_intensity']:
        print(f"     SGi Intensity: {int8_perf['mean_sgi_intensity']:.2f}")
    if int8_perf['mean_p_beta']:
        print(f"     P-beta: {int8_perf['mean_p_beta']:.2f}")
    if int8_perf['mean_frequency']:
        print(f"     Mean Frequency: {int8_perf['mean_frequency']:.2f} Hz")
    if int8_perf['mean_amplitude']:
        print(f"     Mean Amplitude: {int8_perf['mean_amplitude']:.2f} mA")
    
    # Compile results
    results = {
        'architecture': f'{h1}x{h2}',
        'model_timesteps': model_timesteps,
        'size': size_results,
        'fidelity': {
            'mse': fidelity_results['mse'],
            'max_action_diff': fidelity_results['max_action_diff'],
            'mean_action_diff': fidelity_results['mean_action_diff'],
            'std_action_diff': fidelity_results['std_action_diff']
        },
        'inference_time': {
            'fp32': fp32_time,
            'int8': int8_time,
            'speedup': speedup
        },
        'performance': {
            'fp32': fp32_perf,
            'int8': int8_perf
        }
    }
    
    # Add action arrays if available (for plotting)
    if 'fp32_actions' in fidelity_results:
        results['fidelity']['fp32_actions'] = fidelity_results['fp32_actions']
        results['fidelity']['int8_actions'] = fidelity_results['int8_actions']
    
    # Create plots
    print("\n6. Creating comparison plots...")
    create_comparison_plots(results)
    
    # Save results
    output_file = f'quantization_eval_results_{h1}_{h2}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   [OK] Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nModel Size:")
    print(f"  FP32: {size_results['fp32_size_mb']:.4f} MB")
    print(f"  INT8: {size_results['int8_size_mb']:.4f} MB")
    print(f"  Reduction: {size_results['reduction_percent']:.2f}%")
    
    print(f"\nPolicy Fidelity:")
    print(f"  MSE: {fidelity_results['mse']:.6f}")
    print(f"  Mean Action Difference: {fidelity_results['mean_action_diff']:.6f}")
    
    print(f"\nInference Time:")
    print(f"  FP32: {fp32_time['mean_ms']:.4f} ms")
    print(f"  INT8: {int8_time['mean_ms']:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE [OK]")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())

