"""
Windows-compatible power profiling tool for FP32 vs INT8 model comparison.
Uses CPU utilization and inference time as proxies for power consumption.
"""

import argparse
import time
import os
import json
import numpy as np
import torch
import psutil
import threading
from stable_baselines3 import TD3
from BGN_MC import BGN_MC
from torch.ao.quantization import quantize_dynamic

class CPUProfiler:
    """CPU profiler that monitors CPU usage in a separate thread."""
    
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self.running = False
        self.thread = None
        
    def _monitor(self):
        """Monitor CPU usage."""
        process = psutil.Process(os.getpid())
        while self.running:
            cpu_percent = process.cpu_percent(interval=self.interval)
            self.samples.append(cpu_percent)
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if not self.samples:
            return {
                'samples': 0,
                'mean_cpu_percent': None,
                'std_cpu_percent': None,
                'min_cpu_percent': None,
                'max_cpu_percent': None,
                'p50_cpu_percent': None,
                'p90_cpu_percent': None
            }
        
        arr = np.array(self.samples)
        return {
            'samples': int(arr.size),
            'mean_cpu_percent': float(arr.mean()),
            'std_cpu_percent': float(arr.std(ddof=0)),
            'min_cpu_percent': float(arr.min()),
            'max_cpu_percent': float(arr.max()),
            'p50_cpu_percent': float(np.percentile(arr, 50)),
            'p90_cpu_percent': float(np.percentile(arr, 90))
        }


def run_inference(policy, states: np.ndarray, duration_s: int = 60) -> dict:
    """
    Run inference workload for specified duration.
    
    Args:
        policy: Model policy to run
        states: Input states array
        duration_s: Duration in seconds
        
    Returns:
        Dictionary with timing statistics
    """
    policy.eval()
    device = torch.device('cpu')
    torch.set_num_threads(os.cpu_count() or 4)
    
    # Convert states to tensor
    x = torch.from_numpy(states).float().to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = policy(x)
    
    # Measure inference times
    inference_times = []
    t0 = time.time()
    iterations = 0
    
    with torch.no_grad():
        while time.time() - t0 < duration_s:
            iter_start = time.time()
            _ = policy(x)
            iter_end = time.time()
            inference_times.append(iter_end - iter_start)
            iterations += 1
    
    actual_duration = time.time() - t0
    
    if inference_times:
        arr = np.array(inference_times)
        return {
            'duration_s': actual_duration,
            'iterations': iterations,
            'mean_inference_time_ms': float(arr.mean() * 1000),
            'std_inference_time_ms': float(arr.std(ddof=0) * 1000),
            'min_inference_time_ms': float(arr.min() * 1000),
            'max_inference_time_ms': float(arr.max() * 1000),
            'p50_inference_time_ms': float(np.percentile(arr, 50) * 1000),
            'p90_inference_time_ms': float(np.percentile(arr, 90) * 1000),
            'throughput_inf_per_sec': float(iterations / actual_duration)
        }
    else:
        return {
            'duration_s': actual_duration,
            'iterations': 0,
            'mean_inference_time_ms': None,
            'std_inference_time_ms': None
        }


def load_policies(h1=32, h2=32, model_timesteps=2500):
    """
    Load FP32 and INT8 policies.
    
    Args:
        h1: First hidden layer size
        h2: Second hidden layer size
        model_timesteps: Model timesteps to load
        
    Returns:
        Tuple of (fp32_policy, int8_policy)
    """
    model_path = f'models/TD3_{h1}_{h2}/{model_timesteps}.zip'
    qpolicy_path = f'models/policies/qpolicy_{h1}_{h2}.pth'
    
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
    
    return fp32_policy, int8_policy


def collect_calibration_states(num_states=1000):
    """
    Collect calibration states from the environment.
    
    Args:
        num_states: Number of states to collect
        
    Returns:
        numpy array of states
    """
    print(f"Collecting {num_states} calibration states...")
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


def main():
    parser = argparse.ArgumentParser(description='Windows power profiling for FP32 vs INT8 models')
    parser.add_argument('--mode', choices=['fp32', 'int8'], required=True,
                       help='Model mode to profile')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration of profiling in seconds (default: 60)')
    parser.add_argument('--states', type=str, default='states_eval.npy',
                       help='Path to states file (default: states_eval.npy)')
    parser.add_argument('--h1', type=int, default=32,
                       help='First hidden layer size (default: 32)')
    parser.add_argument('--h2', type=int, default=32,
                       help='Second hidden layer size (default: 32)')
    parser.add_argument('--model-timesteps', type=int, default=2500,
                       help='Model timesteps to load (default: 2500)')
    parser.add_argument('--collect-states', action='store_true',
                       help='Collect calibration states if states file not found')
    parser.add_argument('--num-states', type=int, default=1000,
                       help='Number of states to collect (default: 1000)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Windows Power Profiling Tool")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Duration: {args.duration} seconds")
    print(f"Architecture: {args.h1}x{args.h2}")
    print()
    
    # Load or collect states
    if not os.path.exists(args.states):
        if args.collect_states:
            print(f"States file not found. Collecting {args.num_states} states...")
            states = collect_calibration_states(args.num_states)
            np.save(args.states, states)
            print(f"[OK] States saved to {args.states}")
        else:
            print(f"[X] ERROR: States file not found: {args.states}")
            print("   Use --collect-states to collect calibration states")
            return 1
    else:
        print(f"Loading states from {args.states}...")
        states = np.load(args.states)
        print(f"[OK] Loaded {len(states)} states")
    
    # Load policies
    print("\nLoading policies...")
    try:
        fp32_policy, int8_policy = load_policies(
            h1=args.h1, h2=args.h2, model_timesteps=args.model_timesteps
        )
        policy = fp32_policy if args.mode == 'fp32' else int8_policy
        print(f"[OK] {args.mode.upper()} policy loaded")
    except Exception as e:
        print(f"[X] ERROR loading policies: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Start CPU profiler
    print("\nStarting CPU profiler...")
    profiler = CPUProfiler(interval=0.1)
    profiler.start()
    
    # Run inference
    print(f"\nRunning inference for {args.duration} seconds...")
    print("(This may take a while)")
    
    try:
        timing_stats = run_inference(policy, states, args.duration)
    finally:
        cpu_stats = profiler.stop()
    
    # Compile results
    results = {
        'mode': args.mode,
        'architecture': f'{args.h1}x{args.h2}',
        'duration_s': args.duration,
        'cpu_stats': cpu_stats,
        'timing_stats': timing_stats
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    print(f"\nCPU Statistics:")
    if cpu_stats['samples'] > 0:
        print(f"  Samples: {cpu_stats['samples']}")
        print(f"  Mean CPU Usage: {cpu_stats['mean_cpu_percent']:.2f}%")
        print(f"  Std CPU Usage: {cpu_stats['std_cpu_percent']:.2f}%")
        print(f"  Min CPU Usage: {cpu_stats['min_cpu_percent']:.2f}%")
        print(f"  Max CPU Usage: {cpu_stats['max_cpu_percent']:.2f}%")
        print(f"  P50 CPU Usage: {cpu_stats['p50_cpu_percent']:.2f}%")
        print(f"  P90 CPU Usage: {cpu_stats['p90_cpu_percent']:.2f}%")
    else:
        print("  No CPU samples collected")
    
    print(f"\nTiming Statistics:")
    if timing_stats['iterations'] > 0:
        print(f"  Duration: {timing_stats['duration_s']:.2f} seconds")
        print(f"  Iterations: {timing_stats['iterations']}")
        print(f"  Mean Inference Time: {timing_stats['mean_inference_time_ms']:.4f} ms")
        print(f"  Std Inference Time: {timing_stats['std_inference_time_ms']:.4f} ms")
        print(f"  Min Inference Time: {timing_stats['min_inference_time_ms']:.4f} ms")
        print(f"  Max Inference Time: {timing_stats['max_inference_time_ms']:.4f} ms")
        print(f"  P50 Inference Time: {timing_stats['p50_inference_time_ms']:.4f} ms")
        print(f"  P90 Inference Time: {timing_stats['p90_inference_time_ms']:.4f} ms")
        print(f"  Throughput: {timing_stats['throughput_inf_per_sec']:.2f} inferences/sec")
    else:
        print("  No timing data collected")
    
    # Save results to JSON
    output_file = f'power_profile_{args.mode}_{args.h1}_{args.h2}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {output_file}")
    
    print("\n" + "=" * 70)
    return 0


if __name__ == '__main__':
    exit(main())

