"""
Test TD3 model with the working environment.
"""

import numpy as np
import torch
from stable_baselines3 import TD3
from BGN_MC_fixed import BGN_MC_Fixed

def test_td3_with_working_env():
    """Test TD3 model with the fixed environment."""
    print("=== Testing TD3 with Working Environment ===")
    
    # Load environment and model
    env = BGN_MC_Fixed(tmax=1100, pd=True)
    model = TD3.load('models/TD3_64_64/1500.zip', env=env)
    
    # Test model performance
    print("Testing TD3 model performance...")
    episode_rewards = []
    
    for episode in range(3):
        obs, info = env.reset()
        terminated = 0
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while terminated != 1 and step_count < 20:  # Limit steps for testing
            action = model.predict(obs)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if step_count <= 5:  # Show first 5 steps
                print(f"  Step {step_count}: action={action}, reward={reward:.4f}")
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode + 1} total reward: {episode_reward:.4f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"\nAverage reward over {len(episode_rewards)} episodes: {avg_reward:.4f}")
    
    return avg_reward

def test_random_actions():
    """Test random actions for comparison."""
    print("\n=== Testing Random Actions ===")
    
    env = BGN_MC_Fixed(tmax=1100, pd=True)
    episode_rewards = []
    
    for episode in range(3):
        obs, info = env.reset()
        terminated = 0
        episode_reward = 0
        step_count = 0
        
        print(f"\nRandom Episode {episode + 1}:")
        
        while terminated != 1 and step_count < 20:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if step_count <= 5:
                print(f"  Step {step_count}: action={action}, reward={reward:.4f}")
        
        episode_rewards.append(episode_reward)
        print(f"  Random Episode {episode + 1} total reward: {episode_reward:.4f}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"\nAverage random reward over {len(episode_rewards)} episodes: {avg_reward:.4f}")
    
    return avg_reward

def main():
    """Test both TD3 and random actions."""
    print("=== Environment and Model Testing ===")
    
    # Test TD3 model
    td3_reward = test_td3_with_working_env()
    
    # Test random actions
    random_reward = test_random_actions()
    
    print(f"\n=== Results Summary ===")
    print(f"TD3 average reward: {td3_reward:.4f}")
    print(f"Random average reward: {random_reward:.4f}")
    print(f"TD3 improvement: {td3_reward - random_reward:.4f}")
    
    if td3_reward > random_reward:
        print("✅ TD3 model is working better than random!")
    else:
        print("❌ TD3 model needs improvement")
    
    return td3_reward, random_reward

if __name__ == "__main__":
    main()
