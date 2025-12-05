from BGN_MC_Online import BGN_MC_Online
from stable_baselines3 import TD3
import os
import torch

# log things with: tensorboard --logdir=logs/TD3_0 --reload_multifile=True

h1 = 32
h2 = 32

models_dir = f'models/TD3_{h1}_{h2}'
logdir = 'logs'

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(logdir): os.makedirs(logdir)

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[h1, h2], qf=[h1, h2]))

# Use cached mode for smoke test
env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)

model = TD3('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, learning_rate=0.0001)

# model = TD3.load('models/TD3_32/500.zip', env=env)


TIMESTEPS = 500
for i in range(5):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f'{models_dir}/{TIMESTEPS*(i+1)}')