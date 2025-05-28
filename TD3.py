from Matlab_Impl import BGN_MC
import numpy as np
from stable_baselines3 import TD3
import os


models_dir = 'models/TD3'
logdir = 'logs'

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(logdir) :os.makedirs(logdir)

env = BGN_MC.BGN_MC(1200)
env.reset()

model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=logdir, gamma=0.8, learning_rate=0.0001)


model.learn(total_timesteps=1500, reset_num_timesteps=False, tb_log_name='TD3')
model.save(f'{models_dir}/{1500}')