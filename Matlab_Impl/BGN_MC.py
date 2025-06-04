"""
BGN implemention of https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10684783&tag=1
with most computation done in Matlab engine. Wrapped in OpenAI gymnasium env, also does 
state/reward calculation in Python
"""

import gymnasium as gym
import numpy as np
import scipy.io
import matlab.engine
import antropy
eng = matlab.engine.start_matlab()
eng.cd('C:/Users/ncart/Programming/DBS/Matlab_Impl')

class BGN_MC(gym.Env):
    def __init__(self, tmax=1000, pd=True):
        if pd: self.pd = 1
        else: self.pd = 0
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        self.tmax = tmax
        self.past_sgis = 0

    def reset(self, pd=True, seed=None, options=None):
        super().reset(seed=seed)
        eng.bgn_init(self.pd, self.tmax, nargout=0)

        self.smc_spike_times = []
        self.Istim = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['Istim'].flatten()
        for i in range(len(self.Istim)-1): 
            if self.Istim[i] == 0 and self.Istim[i+1] != 0: self.smc_spike_times.append(i+1)
        self.smc_spike_times = np.array(self.smc_spike_times)
        self.past_sgis = 0.5
        observation, reward, terminated, truncated, info = self.step()
        self.past_sgis = info['r3']
        return observation, info

    def step(self, action=np.array([-1.0, -1.0]), sim_time=10000):
        action = np.clip(action, -1, 1)
        freq = 185 * ((action[0] + 1)/2)
        amp = 5000 * ((action[1] + 1)/2)
        
        terminated, _ = eng.bgn_step(freq, amp, sim_time, nargout=2)

        sgis = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['sgis']
        sgis_min = 1082.0999226306508
        sgis_max = 3506.499645178415 
        current_sgis_norm = (np.sum(np.average(np.abs(np.fft.fft(sgis)), axis=0)[1:20]) - sgis_min)/(sgis_max-sgis_min)
        r1_min = 0
        r1_max = 1
        r1 = ((self.past_sgis - current_sgis_norm)-r1_min)/(r1_max-r1_min)

        r2_max = 1
        # r2 = np.sqrt((freq/10 * amp * 0.000030))/r2_max
        theta = 0.85
        r2 = (theta * (freq/185) + (1-theta) * amp/5000)/r2_max
 
        r3 = current_sgis_norm

        r4 = 1 if r1==0 and r2==0 and r3==0 else 0

        self.past_sgis = current_sgis_norm

        epsilon = 0.68
        reward = epsilon * -r3 + (1-epsilon) * -r2

        vgi = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['vgi']
        vsn = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['vsn']
        i = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['i'].flatten()[0]

        sd_min = 0.07085760036983213
        sd_max = 0.2060141359064851
        sd = (np.average(np.std(sgis, axis=1)) - sd_min)/(sd_max-sd_min)

        A_min = 0.0050925265308826816
        A_max = 0.042526051971444476
        A = np.average(np.var(sgis, axis=1))
        A_norm = (A-A_min)/(A_max-A_min)

        M_min = 0.002065105439508318
        M_max = 0.04131553203125676
        M = np.average(np.sqrt(np.var(np.diff(sgis), axis=1)/A))
        M_norm = (M-M_min)/(M_max-M_min)

        C_min = 0.10438751644244766
        C_max = 0.993985189386497
        C = np.average(np.sqrt(np.var(np.diff(np.diff(sgis)), axis=1)/A) / M)
        C_norm = (C-C_min)/(C_max-C_min)

        Pb_min = 4547158.333768198  
        Pb_max = 5801474.044129377
        Pb = (np.sum(np.average(np.abs(np.fft.fft(vgi[:, i-sim_time:i])) / 0.1, axis=0)[12:31]) - Pb_min)/(Pb_max - Pb_min) # 0.1 bc env step is 1/10 of a s

        SampEn_min = 0.001344649634141824 
        SampEn_max = 0.005891566645584552
        SampEn = (np.average([antropy.sample_entropy(vsn[n][i-sim_time:i]) for n in range(10)]) - SampEn_min)/(SampEn_max - SampEn_min)
        observation = np.array((sd, A_norm, M_norm, C_norm, Pb, SampEn))

        return observation, reward, bool(terminated), False, {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4}
    

    def calculate_ei(self):
        vth = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['vth']
        i = scipy.io.loadmat('C:/Users/ncart/Programming/DBS/Matlab_Impl/bgn_vars.mat')['i'].flatten()[0]

        pulses = [time for time in self.smc_spike_times if time > i-10000 and time < i]
        e = 0
        for n in range(10):
            spikes = []
            used_spikes = set()
            for t in range(10000-2):
                if vth[n][t + i-10000] < -40 and vth[n][t+1 + i-10000] >= -40:
                    spikes.append(t + i-10000) 
            for pulse in pulses:
                valid_spikes = [i for i, s in enumerate(spikes) if pulse <= s <= pulse + 2500 and i not in used_spikes]
                if len(valid_spikes) == 0: e += 1
                elif len(valid_spikes) > 1: 
                    e += 1
                    used_spikes.add(valid_spikes[0])
                else: 
                    used_spikes.add(valid_spikes[0])
            extra_spikes = set(range(len(spikes))) - used_spikes
            e += len(extra_spikes)
        if len(pulses) == 0: return 0
        return e/(10*len(pulses))
