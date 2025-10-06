"""
Collect calibration states from FP32 TD3 actor for quantization.
Saves states to states_eval.npy for later use in quantization.
"""

import numpy as np
import torch
from stable_baselines3 import TD3
from BGN_MC_no_matlab import BGN_MC_NoMatlab
import os

def collect_calibration_states(num_states=10000, model_path='models/TD3_64_64/1500.zip'):
    """
    Collect calibration states by running the FP32 actor on the environment.
    
    Args:
        num_states: Number of states to collect
        model_path: Path to the trained TD3 model
    
    Returns:
        states: Array of shape (num_states, 6) containing observation states
    """
    print(f"Collecting {num_states} calibration states...")
    
    # Load environment and model
    env = BGN_MC_NoMatlab(tmax=1100, pd=True)
    model = TD3.load(model_path, env=env)
    
    states = []
    episodes = 0
    
    while len(states) < num_states:
        obs, info = env.reset()
        terminated = 0
        episode_states = []
        
        while terminated != 1 and len(states) < num_states:
            # Store the current state
            states.append(obs.copy())
            episode_states.append(obs.copy())
            
            # Get action from model
            action = model.predict(obs)[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
        
        episodes += 1
        print(f"Episode {episodes}: Collected {len(episode_states)} states (Total: {len(states)})")
    
    # Convert to numpy array and truncate to exact number requested
    states = np.array(states[:num_states])
    
    print(f"Collected {len(states)} states from {episodes} episodes")
    print(f"State shape: {states.shape}")
    print(f"State range: [{states.min():.4f}, {states.max():.4f}]")
    
    return states

def main():
    # Collect states
    states = collect_calibration_states(num_states=10000)
    
    # Save states
    np.save('states_eval.npy', states)
    print(f"Saved {len(states)} states to states_eval.npy")
    
    # Verify saved file
    loaded_states = np.load('states_eval.npy')
    print(f"Verification: Loaded {len(loaded_states)} states with shape {loaded_states.shape}")

if __name__ == "__main__":
    main()
