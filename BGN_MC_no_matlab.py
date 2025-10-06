"""
Modified BGN implementation for quantization testing without MATLAB dependency.
Uses pre-computed bgn_vars.mat data for demonstration purposes.
"""

import gymnasium as gym
import numpy as np
import scipy.io
import antropy

class BGN_MC_NoMatlab(gym.Env):
    def __init__(self, tmax=1000, pd=True):
        if pd: 
            self.pd = 1
        else: 
            self.pd = 0
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        self.tmax = tmax
        self.past_sgis = 0
        self.current_step = 0
        self.max_steps = tmax // 100  # Assuming 100ms per step
        
        # Load pre-computed data
        self.bgn_data = scipy.io.loadmat('bgn_vars.mat')
        self.sgis = self.bgn_data['sgis']
        self.vgi = self.bgn_data['vgi']
        self.vth = self.bgn_data['vth']
        self.vsn = self.bgn_data['vsn']
        self.vge = self.bgn_data['vge']
        self.Istim = self.bgn_data['Istim'].flatten()

    def reset(self, pd=True, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.past_sgis = 0.5
        
        # Simulate initial observation
        observation, reward, terminated, truncated, info = self.step()
        self.past_sgis = info['r3']
        return observation, info

    def step(self, action=np.array([-1.0, -1.0]), sim_time=10000):
        action = np.clip(action, -1, 1)
        freq = 185 * ((action[0] + 1)/2)
        amp = 5000 * ((action[1] + 1)/2)
        
        self.current_step += 1
        terminated = 1 if self.current_step >= self.max_steps else 0
        
        # Use pre-computed data for demonstration
        # In real implementation, this would call MATLAB engine
        sgis = self.sgis
        sgis_min = 1082.0999226306508
        sgis_max = 3506.499645178415 
        current_sgis_norm = (np.sum(np.average(np.abs(np.fft.fft(sgis)), axis=0)[1:20]) - sgis_min)/(sgis_max-sgis_min)
        
        # Calculate reward based on SGi reduction
        reward = self.past_sgis - current_sgis_norm
        self.past_sgis = current_sgis_norm
        
        # Create observation vector (6 elements as per original)
        observation = np.array([
            current_sgis_norm,
            freq / 185.0,  # Normalized frequency
            amp / 5000.0,  # Normalized amplitude
            np.mean(self.vgi),  # Average GPi voltage
            np.mean(self.vth),  # Average thalamus voltage
            np.mean(self.vsn)   # Average STN voltage
        ], dtype=np.float64)
        
        info = {
            'r1': current_sgis_norm,
            'r2': freq,
            'r3': current_sgis_norm,
            'terminated': terminated
        }
        
        truncated = 0
        return observation, reward, terminated, truncated, info
