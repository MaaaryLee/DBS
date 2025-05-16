"""
BGN implemented with most computation done in Matlab engine.
Wrapped in OpenAI gymnasium env, also does state/reward calculation
in Python
"""

import gymnasium as gym
import numpy as np
import scipy.io
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('C:/Users/ncart/Programming/DBS/Matlab_Impl')

class BGN_MC(gym.Env):
    def __init__(self, tmax=1000):
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        self.tmax = tmax
        self.past_sgi = 0
        # self.action_list = [0, 100, 115, 130, 145, 160, 175, 190]

    def reset(self, seed=None, options=None):
        self.pd = np.random.uniform(0.7, 1.0)
        super().reset(seed=seed)
        eng.bgn_init(self.pd, self.tmax, nargout=0)
        observation, reward, terminated, truncated, info = self.step()
        return observation, info

    def step(self, action=np.array([-1.0, -1.0]), sim_time=20000):
        action = np.clip(action, -1, 1)
        freq = 100 * action[0] + 100 # maps [-1,1] --> [0, 200]
        amp = 2500 * action[1] + 2500 # maps [-1,1] --> [0, 5000]
        
        terminated, _ = eng.bgn_step(freq, amp, sim_time, nargout=2)

        sgis = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['sgis']
        sgis_fft_total = np.sum(np.average(np.abs(np.fft.fft(sgis)), axis=0)[0:50])
        sgis_fft_total_norm = (sgis_fft_total-12878)/(16541-12878) # derived from min-max normalization
      
        # vth = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['vth']

        # r1 = -amp*30*int(freq/(1000/(sim_time/100)))/6000000 # based on min-max norm, change '300' and the max when modulating amplitude
        r1 = -np.sqrt(1/sim_time * amp*amp * freq/(1000/(sim_time/100)))/224
        r2 = -sgis_fft_total_norm
        alpha = 0.3
        reward = alpha * r1 + (1-alpha) * r2

        A = np.average(np.var(sgis, axis=1))
        M = np.average(np.sqrt(np.var(np.diff(sgis), axis=1)/A))
        C = np.average(np.sqrt(np.var(np.diff(np.diff(sgis)), axis=1)/A) / M)
        # SampEn = np.average([antropy.sample_entropy(sgis[i]) for i in range(10)])
        observation = np.array((np.average(np.std(sgis, axis=1)), A, M, C, sgis_fft_total_norm))

        return observation, reward, bool(terminated), False, {'r1': r1, 'r2': r2, 'alpha': alpha}