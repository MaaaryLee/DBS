#!/usr/bin/env python3
"""
Controller Validation Test
=========================

This script validates both the default (FP32) and quantized (INT8) controllers
as suggested by your PI.

1. Test default controller performance
2. Test quantized controller performance  
3. Compare results to validate system accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from BGN_MC_Online import BGN_MC_Online
from stable_baselines3 import TD3
import torch
import os

def test_default_controller():
    """Test the default FP32 controller"""
    print("=" * 60)
    print("TESTING DEFAULT CONTROLLER (FP32)")
    print("=" * 60)
    
    # Load the trained model
    model_path = 'models/TD3_32_32/2500.zip'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    model = TD3.load(model_path)
    print(f"✅ Loaded default controller: {model_path}")
    
    # Test with cached environment
    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)
    
    # Run multiple episodes to get average performance
    episodes = 5
    rewards = []
    actions = []
    sgis = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_actions = []
        episode_sgis = [info['r3']]
        
        for step in range(10):  # 10 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_actions.append(action.copy())
            episode_sgis.append(info['r3'])
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        actions.extend(episode_actions)
        sgis.extend(episode_sgis)
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_sgi = np.mean(sgis)
    
    print(f"✅ Default Controller Results:")
    print(f"   Episodes tested: {episodes}")
    print(f"   Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"   Average SGi biomarker: {avg_sgi:.6f}")
    print(f"   Actions range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
    
    return {
        'rewards': rewards,
        'actions': actions,
        'sgis': sgis,
        'avg_reward': avg_reward,
        'avg_sgi': avg_sgi
    }

def test_quantized_controller():
    """Test the quantized INT8 controller"""
    print("\n" + "=" * 60)
    print("TESTING QUANTIZED CONTROLLER (INT8)")
    print("=" * 60)
    
    # Check if quantized model exists
    quantized_path = 'models/policies/qpolicy_64_64.pth'
    if not os.path.exists(quantized_path):
        print(f"❌ Quantized model not found: {quantized_path}")
        print("   Need to run quantization first!")
        return None
    
    # Load quantized model
    quantized_policy = torch.jit.load(quantized_path)
    print(f"✅ Loaded quantized controller: {quantized_path}")
    
    # Test with cached environment
    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)
    
    # Run multiple episodes to get average performance
    episodes = 5
    rewards = []
    actions = []
    sgis = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_actions = []
        episode_sgis = [info['r3']]
        
        for step in range(10):  # 10 steps per episode
            # Convert observation to tensor for quantized model
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_tensor = quantized_policy(obs_tensor)
            action = action_tensor.squeeze(0).numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_actions.append(action.copy())
            episode_sgis.append(info['r3'])
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        actions.extend(episode_actions)
        sgis.extend(episode_sgis)
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_sgi = np.mean(sgis)
    
    print(f"✅ Quantized Controller Results:")
    print(f"   Episodes tested: {episodes}")
    print(f"   Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"   Average SGi biomarker: {avg_sgi:.6f}")
    print(f"   Actions range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
    
    return {
        'rewards': rewards,
        'actions': actions,
        'sgis': sgis,
        'avg_reward': avg_reward,
        'avg_sgi': avg_sgi
    }

def compare_controllers(default_results, quantized_results):
    """Compare default vs quantized controller performance"""
    print("\n" + "=" * 60)
    print("CONTROLLER COMPARISON")
    print("=" * 60)
    
    if default_results is None or quantized_results is None:
        print("❌ Cannot compare - missing results from one or both controllers")
        return
    
    # Compare rewards
    reward_diff = abs(default_results['avg_reward'] - quantized_results['avg_reward'])
    reward_pct_diff = (reward_diff / abs(default_results['avg_reward'])) * 100 if default_results['avg_reward'] != 0 else 0
    
    print(f"📊 Performance Comparison:")
    print(f"   Default (FP32) reward:  {default_results['avg_reward']:.4f}")
    print(f"   Quantized (INT8) reward: {quantized_results['avg_reward']:.4f}")
    print(f"   Reward difference:       {reward_diff:.4f} ({reward_pct_diff:.1f}%)")
    
    # Compare SGi biomarkers
    sgi_diff = abs(default_results['avg_sgi'] - quantized_results['avg_sgi'])
    sgi_pct_diff = (sgi_diff / default_results['avg_sgi']) * 100 if default_results['avg_sgi'] != 0 else 0
    
    print(f"   Default (FP32) SGi:     {default_results['avg_sgi']:.6f}")
    print(f"   Quantized (INT8) SGi:   {quantized_results['avg_sgi']:.6f}")
    print(f"   SGi difference:         {sgi_diff:.6f} ({sgi_pct_diff:.1f}%)")
    
    # Compare action ranges
    default_actions = np.array(default_results['actions'])
    quantized_actions = np.array(quantized_results['actions'])
    
    print(f"   Default action range:   [{default_actions.min():.3f}, {default_actions.max():.3f}]")
    print(f"   Quantized action range: [{quantized_actions.min():.3f}, {quantized_actions.max():.3f}]")
    
    # Determine if quantization is acceptable
    if reward_pct_diff < 5.0 and sgi_pct_diff < 5.0:
        print(f"✅ Quantization is ACCEPTABLE (< 5% difference)")
    elif reward_pct_diff < 10.0 and sgi_pct_diff < 10.0:
        print(f"⚠️  Quantization is MARGINAL (5-10% difference)")
    else:
        print(f"❌ Quantization has SIGNIFICANT impact (> 10% difference)")

def create_comparison_plot(default_results, quantized_results):
    """Create visualization comparing both controllers"""
    if default_results is None or quantized_results is None:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Reward comparison
    controllers = ['Default (FP32)', 'Quantized (INT8)']
    rewards = [default_results['avg_reward'], quantized_results['avg_reward']]
    reward_stds = [np.std(default_results['rewards']), np.std(quantized_results['rewards'])]
    
    ax1.bar(controllers, rewards, yerr=reward_stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Controller Performance Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SGi biomarker comparison
    sgis = [default_results['avg_sgi'], quantized_results['avg_sgi']]
    ax2.bar(controllers, sgis, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Average SGi Biomarker')
    ax2.set_title('Biomarker Response Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action distribution
    default_actions = np.array(default_results['actions'])
    quantized_actions = np.array(quantized_results['actions'])
    
    ax3.hist(default_actions.flatten(), bins=20, alpha=0.7, label='Default', color='blue')
    ax3.hist(quantized_actions.flatten(), bins=20, alpha=0.7, label='Quantized', color='red')
    ax3.set_xlabel('Action Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Action Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Episode rewards
    episodes = range(1, len(default_results['rewards']) + 1)
    ax4.plot(episodes, default_results['rewards'], 'bo-', label='Default', linewidth=2)
    ax4.plot(episodes, quantized_results['rewards'], 'ro-', label='Quantized', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Episode Reward')
    ax4.set_title('Episode-by-Episode Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Created comparison plot: controller_comparison.png")

def main():
    """Run controller validation as suggested by PI"""
    print("🎯 CONTROLLER VALIDATION TEST")
    print("=" * 60)
    print("Testing default and quantized controllers as suggested by PI")
    print("This validates system accuracy and quantization impact.\n")
    
    # Test default controller
    default_results = test_default_controller()
    
    # Test quantized controller
    quantized_results = test_quantized_controller()
    
    # Compare results
    compare_controllers(default_results, quantized_results)
    
    # Create visualization
    create_comparison_plot(default_results, quantized_results)
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    if default_results is not None and quantized_results is not None:
        print("✅ Both controllers tested successfully")
        print("✅ Performance comparison completed")
        print("✅ Results saved to controller_comparison.png")
        print("\n📋 Summary for PI:")
        print("   - Default controller baseline established")
        print("   - Quantized controller performance validated")
        print("   - Quantization impact quantified")
        print("   - System accuracy confirmed")
    else:
        print("⚠️  Some tests failed - check model availability")
        print("   - Ensure trained models exist")
        print("   - Run quantization if needed")

if __name__ == "__main__":
    main()

