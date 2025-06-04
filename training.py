from Matlab_Impl import BGN_MC
import numpy as np
from stable_baselines3 import SAC
import os


models_dir = 'models/SAC_SGi_monophase'
logdir = 'logs'

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(logdir): os.makedirs(logdir)

env = BGN_MC.BGN_MC(tmax=1100)

model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# model = RecurrentPPO.load('models/RPPO_SGi_biphase/1000.zip', env=env)


TIMESTEPS = 500
for i in range(10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='SAC')
    model.save(f'{models_dir}/{TIMESTEPS*(i+1)}')