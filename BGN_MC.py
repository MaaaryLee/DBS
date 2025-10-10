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
eng.cd('C:/Users/ncart/Programming/PerCom2025')

class BGN_MC(gym.Env):
    def __init__(self, mode, tmax=1000, pd=True):
        self.mode = mode
        if pd: self.pd = 1
        else: self.pd = 0
        
        if mode == 'hvgi': 
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        elif mode == 'hvgi_sgi':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        elif mode == 'hsgi':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)


        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        self.tmax = tmax
        self.past_sgis = 0

    def reset(self, pd=True, seed=None, options=None):
        super().reset(seed=seed)
        eng.bgn_init(self.pd, self.tmax, nargout=0)

        self.smc_spike_times = []
        self.Istim = scipy.io.loadmat('bgn_vars.mat')['Istim'].flatten()
        for i in range(len(self.Istim)-1): 
            if self.Istim[i] == 0 and self.Istim[i+1] != 0: self.smc_spike_times.append(i+1)
        self.smc_spike_times = np.array(self.smc_spike_times)
        observation, reward, terminated, truncated, info = self.step()
        return observation, info

    def step(self, action=np.array([-1.0, -1.0]), sim_time=10000):
        action = np.clip(action, -1, 1)
        freq = 185 * ((action[0] + 1)/2)
        amp = 5000 * ((action[1] + 1)/2)
        
        terminated, vgi_last = eng.bgn_step(freq, amp, sim_time, nargout=2)
        sgis = scipy.io.loadmat('bgn_vars.mat')['sgis']
        sgis_min = 1082.0999226306508
        sgis_max = 3506.499645178415 
        current_sgis_norm = (np.sum(np.average(np.abs(np.fft.fft(sgis)), axis=0)[1:20]) - sgis_min)/(sgis_max-sgis_min)
  

        r2_max = 1
        theta = 0.85
        r2 = (theta * (freq/185) + (1-theta) * amp/5000)/r2_max
 
        r3 = current_sgis_norm

        epsilon = 0.68
        reward = epsilon * -r3 + (1-epsilon) * -r2

        # vgi = scipy.io.loadmat('bgn_vars.mat')['vgi']
        
        i = scipy.io.loadmat('bgn_vars.mat')['i'].flatten()[0]
        observation = None

        # 4 ELEMENT STATE REPRESENTATION BASED ON THE HJORTH OF VGI SIGNAL
        if self.mode == 'hvgi':
            vgi = scipy.io.loadmat('bgn_vars.mat')['vgi']
            vgi = vgi[:, i-sim_time:i:40]

            # sd_min = 25.849944778332663  
            # sd_max = 32.36659864207871
            sd_min = 25.835764199283012 
            sd_max = 31.536312171457585
            sd = (np.average(np.std(vgi, axis=1)) - sd_min)/(sd_max-sd_min)

            # A_min = 672.2282360031815  
            # A_max = 1047.66991012089
            A_min = 670.8070448523671 
            A_max = 912.5109313814773
            A = np.average(np.var(vgi, axis=1))
            A_norm = (A-A_min)/(A_max-A_min)

            # M_min = 0.09201175246694249 
            # M_max = 0.09689537936008763
            M_min = 0.804377826050485  
            M_max = 0.8493461351479681
            M = np.average(np.sqrt(np.var(np.diff(vgi), axis=1)/A))
            M_norm = (M-M_min)/(M_max-M_min)

            # C_min = 0.7911088367808019  
            # C_max = 0.8618411558467868
            C_min = 1.3447664357706237  
            C_max = 1.373982618086779
            C = np.average(np.sqrt(np.var(np.diff(np.diff(vgi)), axis=1)/A) / M)
            C_norm = (C-C_min)/(C_max-C_min)

            observation = np.array((sd, A_norm, M_norm, C_norm))

        # 4 ELEMEMENT STATE REPRESENTATION BASED ON SGI SIGNAL HJORTH
        if self.mode == 'hsgi':
            sgis = sgis[:, 0:-1:2]

            sd_min = 0.05245132763338163 
            sd_max = 0.19762883444956122
            sd = (np.average(np.std(sgis, axis=1)) - sd_min)/(sd_max-sd_min)

            A_min = 0.0029166595480043978  
            A_max = 0.040595996082619296
            A = np.average(np.var(sgis, axis=1))
            A_norm = (A-A_min)/(A_max-A_min)

            M_min = 0.004055228611268984 
            M_max = 0.016844722932332877
            M = np.average(np.sqrt(np.var(np.diff(sgis), axis=1)/A))
            M_norm = (M-M_min)/(M_max-M_min)

            C_min = 0.12272551211558369 
            C_max = 0.1602778903071769
            C = np.average(np.sqrt(np.var(np.diff(np.diff(sgis)), axis=1)/A) / M)
            C_norm = (C-C_min)/(C_max-C_min)

            observation = np.array((sd, A_norm, M_norm, C_norm))


        # BASIC 6 ELEMENT STATE REPRESENTATION
        if self.mode == 'hvgi_sgi':
            vgi = scipy.io.loadmat('bgn_vars.mat')['vgi']
            vsn = scipy.io.loadmat('bgn_vars.mat')['vsn']
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
            Pb = (np.sum(np.average(np.abs(np.fft.fft(vgi[:, i-sim_time:i:2])) / 0.1, axis=0)[12:31]) - Pb_min)/(Pb_max - Pb_min) # 0.1 bc env step is 1/10 of a s

            SampEn_min = 0.001344649634141824 
            SampEn_max = 0.005891566645584552
            SampEn = (np.average([antropy.sample_entropy(vsn[n][i-sim_time:i:2]) for n in range(10)]) - SampEn_min)/(SampEn_max - SampEn_min)
            observation = np.array((sd, A_norm, M_norm, C_norm, Pb, SampEn))

        return observation, reward, bool(terminated), False, {'r2': r2, 'r3': r3}