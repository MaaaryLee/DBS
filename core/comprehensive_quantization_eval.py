"""
Comprehensive quantization evaluation script.
Compares FP32 vs INT8 actor modules on accuracy, size, performance, and power metrics.
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from sklearn.metrics import mean_squared_error

import quantize_model  # noqa: F401 (register TD3Actor classes)

VARIANT_SUFFIX = {
    "dynamic_int8": "dynamic",
    "static_int8": "static",
}


def _create_env(tmax=1100, pd=True):
    from BGN_MC import BGN_MC

    return BGN_MC(tmax=tmax, pd=pd)


def load_models(h1=32, h2=32, variant="static_int8", with_env=True):
    """
    Load FP32 and quantized actors.

    Returns:
        Tuple of (fp32_actor, int8_actor, env)
    """
    import sys

    # Ensure pickled modules saved under __main__ can be deserialized.
    sys.modules['__main__'].TD3Actor = quantize_model.TD3Actor
    sys.modules['__main__'].QuantizableTD3Actor = quantize_model.QuantizableTD3Actor

    suffix = VARIANT_SUFFIX[variant]
    fp32_path = Path(f"models/policies/actor_fp32_{h1}_{h2}.pt")
    int8_path = Path(f"models/policies/actor_int8_{suffix}_{h1}_{h2}.pt")

    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 actor checkpoint not found: {fp32_path}")
    if not int8_path.exists():
        raise FileNotFoundError(f"Quantized actor checkpoint not found: {int8_path}")

    fp32_actor = torch.load(fp32_path, map_location="cpu")
    fp32_actor.eval()

    int8_actor = torch.load(int8_path, map_location="cpu")
    int8_actor.eval()

    env = _create_env(tmax=1100, pd=True) if with_env else None
    return fp32_actor, int8_actor, env


def measure_model_size(h1=32, h2=32, variant="static_int8"):
    """Measure model file sizes."""
    suffix = VARIANT_SUFFIX[variant]
    policy_path = Path(f'models/policies/actor_fp32_{h1}_{h2}.pt')
    qpolicy_path = Path(f'models/policies/actor_int8_{suffix}_{h1}_{h2}.pt')
    
    fp32_size = policy_path.stat().st_size / (1024 * 1024)  # MB
    int8_size = qpolicy_path.stat().st_size / (1024 * 1024)  # MB
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
    env = _create_env(tmax=1100, pd=True)
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


def measure_inference_time(
    policy,
    states_path='states_eval.npy',
    num_runs=100,
    use_quantized_core=False,
):
    """
    Measure inference time.
    
    Returns:
        Dictionary with timing statistics
    """
    mode = 'full'
    if not os.path.exists(states_path):
        states = collect_calibration_states()
        np.save(states_path, states)
    else:
        states = np.load(states_path)
    
    states_tensor = torch.from_numpy(states[:100]).float()  # Use subset for timing

    run_module = policy
    sample = states_tensor

    if use_quantized_core and hasattr(policy, "backbone"):
        backbone = getattr(policy, "backbone", None)
        first_layer = None
        if hasattr(backbone, "__getitem__"):
            try:
                first_layer = backbone[0]
            except (IndexError, TypeError):
                first_layer = None
        if (
            backbone is not None
            and first_layer is not None
            and hasattr(first_layer, "scale")
            and hasattr(first_layer, "zero_point")
        ):
            sample = torch.quantize_per_tensor(
                states_tensor,
                scale=float(first_layer.scale),
                zero_point=int(first_layer.zero_point),
                dtype=torch.quint8,
            )
            run_module = backbone
            mode = 'backbone'

    # Warm up
    with torch.inference_mode():
        for _ in range(10):
            run_module(sample)
    
    # Measure inference time
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        with torch.inference_mode():
            run_module(sample)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    arr = np.array(times)
    return {
        'mean_ms': float(arr.mean() * 1000),
        'std_ms': float(arr.std(ddof=0) * 1000),
        'min_ms': float(arr.min() * 1000),
        'max_ms': float(arr.max() * 1000),
        'p50_ms': float(np.percentile(arr, 50) * 1000),
        'p90_ms': float(np.percentile(arr, 90) * 1000),
        'mode': mode,
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
            env = _create_env(tmax=1100, pd=True)
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
    # Avoid blocking CLI runs by not calling plt.show().
    plt.close(fig)
    
    # 2. Inference time comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    fp32_time_data = results['inference_time']['fp32']
    int8_time_data = results['inference_time']['int8']
    fp32_time = fp32_time_data['mean_ms']
    int8_time = int8_time_data['mean_ms']
    int8_label = 'INT8 (core)' if int8_time_data.get('mode') == 'backbone' else 'INT8'
    times = [fp32_time, int8_time]
    bars = ax.bar([labels[0], int8_label], times, color=colors)
    ax.set_ylabel('Mean Inference Time (ms)', fontsize=12)
    title = 'Inference Time Comparison'
    if int8_time_data.get('mode') == 'backbone':
        title += ' (pre-quantized inputs)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f} ms',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/inference_time_comparison.png', dpi=150)
    # Avoid blocking CLI runs by not calling plt.show().
    plt.close(fig)
    
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
        # Avoid blocking CLI runs by not calling plt.show().
        plt.close(fig)
    
    print(f"[OK] Plots saved to {output_dir}/")


def main(
    variant="static_int8",
    num_episodes=5,
    output_dir="quantization_eval_latest",
    skip_env=False,
    h1=22,
    h2=22,
):
    print("=" * 70)
    print("Comprehensive Quantization Evaluation")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n1. Loading models...")
    try:
        fp32_policy, int8_policy, env = load_models(
            h1=h1, h2=h2, variant=variant, with_env=not skip_env
        )
        print("   [OK] Models loaded successfully")
    except Exception as e:
        print(f"   [X] ERROR loading models: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Measure model sizes
    print("\n2. Measuring model sizes...")
    size_results = measure_model_size(h1=h1, h2=h2, variant=variant)
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
    print(f"   FP32 mean: {fp32_time['mean_ms']:.4f} ms +/- {fp32_time['std_ms']:.4f} ms")
    int8_time = measure_inference_time(int8_policy, use_quantized_core=True)
    int8_mode = int8_time.get('mode', 'full')
    mode_desc = "core-only (pre-quantized inputs)" if int8_mode == 'backbone' else "with quant/dequant stubs"
    print(f"   INT8 mean: {int8_time['mean_ms']:.4f} ms +/- {int8_time['std_ms']:.4f} ms ({mode_desc})")
    speedup = fp32_time['mean_ms'] / int8_time['mean_ms'] if int8_time['mean_ms'] > 0 else 0
    print(f"   Speedup ({mode_desc}): {speedup:.2f}x")
    
    # Evaluate performance
    if skip_env:
        print("\n5. Skipping environment performance evaluation (--skip-env).")
        fp32_perf = None
        int8_perf = None
    else:
        print(f"\n5. Evaluating performance ({num_episodes} episodes)...")
        print("   FP32 model...")
        fp32_perf = evaluate_performance(
            fp32_policy, env, num_episodes=num_episodes, model_name="FP32"
        )
        print("   INT8 model...")
        int8_perf = evaluate_performance(
            int8_policy, env, num_episodes=num_episodes, model_name="INT8"
        )
        
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
        } if not skip_env else None
    }
    # Add action arrays if available (for plotting)
    if 'fp32_actions' in fidelity_results:
        results['fidelity']['fp32_actions'] = fidelity_results['fp32_actions']
        results['fidelity']['int8_actions'] = fidelity_results['int8_actions']
    
    # Create plots
    print("\n6. Creating comparison plots...")
    create_comparison_plots(results, output_dir=str(plots_dir))
    
    # Save results
    output_file = output_dir / f'quantization_eval_results_{variant}_{h1}_{h2}.json'
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
    mode_label = "core-only (pre-quantized inputs)" if int8_mode == 'backbone' else "with quant/dequant stubs"
    print(f"  INT8: {int8_time['mean_ms']:.4f} ms ({mode_label})")
    print(f"  Speedup ({mode_label}): {speedup:.2f}x")

    if not skip_env:
        print(f"\nEnvironment Performance:")
        if fp32_perf['mean_sgi_intensity']:
            print(f"  FP32 SGi Intensity: {fp32_perf['mean_sgi_intensity']:.2f}")
        if int8_perf['mean_sgi_intensity']:
            print(f"  INT8 SGi Intensity: {int8_perf['mean_sgi_intensity']:.2f}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE [OK]")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive TD3 actor quantization evaluation")
    parser.add_argument('--variant', choices=['dynamic_int8', 'static_int8'], default='static_int8')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes for environment rollouts')
    parser.add_argument('--output-dir', type=str, default='quantization_eval_latest', help='Directory to store results and plots')
    parser.add_argument('--skip-env', action='store_true', help='Skip MATLAB-dependent environment rollouts')
    parser.add_argument('--h1', type=int, default=22, help='First hidden layer size')
    parser.add_argument('--h2', type=int, default=22, help='Second hidden layer size')
    args = parser.parse_args()
    exit(
        main(
            variant=args.variant,
            num_episodes=args.episodes,
            output_dir=args.output_dir,
            skip_env=args.skip_env,
            h1=args.h1,
            h2=args.h2,
        )
    )

