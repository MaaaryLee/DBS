"""
BGN implemented with most computation done in Matlab engine.
Wrapped in OpenAI gymnasium env, also does state/reward calculation
in Python
"""

import gymnasium as gym
import numpy as np
import scipy.io
import antropy
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('C:/Users/ncart/Programming/DBS/Matlab_Impl')

class BGN_M(gym.Env):
    def __init__(self, pd, tmax=1000):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.action_space = gym.spaces.Discrete(8)

        self.pd = pd
        self.tmax = tmax
        self.past_sgi = 0
        self.action_list = [0, 100, 115, 130, 145, 160, 175, 190]

    def reset(self, seed=None, options=None):
        self.seed = seed
        super().reset(seed=seed)
        eng.bgn_init(self.pd, self.tmax, np.random.randint(9999999) if seed==None else seed, nargout=0)
        observation, reward, terminated, truncated, info = self.step()
        return observation, info

    def step(self, action=0, sim_time=20000):
        terminated, _ = eng.bgn_step(action, sim_time, nargout=2)

        sgis = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['sgis']
        sgis_fft_total = np.sum(np.average(np.abs(np.fft.fft(sgis)), axis=0)[0:50])
        sgis_fft_total_norm = (sgis_fft_total-12878)/(16541-12878) # derived from min-max normalization
      
        r1 = -300*30*int(self.action_list[action]/(1000/(sim_time/100)))/342000 # based on min-max norm, change '300' and the max when modulating amplitude
        r2 = -sgis_fft_total_norm
        alpha = 0.2
        reward = alpha * r1 + (1-alpha) * r2

        A = np.average(np.var(sgis, axis=1))
        M = np.average(np.sqrt(np.var(np.diff(sgis), axis=1)/A))
        C = np.average(np.sqrt(np.var(np.diff(np.diff(sgis)), axis=1)/A) / M)
        # SampEn = np.average([antropy.sample_entropy(sgis[i]) for i in range(10)])
        observation = np.array((np.average(np.std(sgis, axis=1)), A, M, C, sgis_fft_total_norm))

        return observation, reward, terminated, False, None