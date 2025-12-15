import argparse
import os
import time

import torch
from stable_baselines3 import TD3

from BGN_MC_Online import BGN_MC_Online

# log things with: tensorboard --logdir=logs/TD3_0 --reload_multifile=True

DEFAULT_H1 = 22
DEFAULT_H2 = 22

parser = argparse.ArgumentParser(description="Train TD3 policy (default: 22x22).")
parser.add_argument("--h1", type=int, default=DEFAULT_H1, help="First hidden layer size")
parser.add_argument("--h2", type=int, default=DEFAULT_H2, help="Second hidden layer size")
parser.add_argument("--timesteps", type=int, default=500, help="Timesteps per checkpoint save")
parser.add_argument("--checkpoints", type=int, default=5, help="Number of checkpoints to save")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--tb-log-name",
    type=str,
    default=None,
    help="TensorBoard run name (default: TD3_<h1>_<h2>_<timestamp>)",
)
args = parser.parse_args()

h1 = args.h1
h2 = args.h2

models_dir = f'models/TD3_{h1}_{h2}'
logdir = 'logs'

if not os.path.exists(models_dir): os.makedirs(models_dir)
if not os.path.exists(logdir): os.makedirs(logdir)

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[h1, h2], qf=[h1, h2]))

# Use cached mode for smoke test
env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)

tb_log_name = args.tb_log_name
if not tb_log_name:
    tb_log_name = f"TD3_{h1}_{h2}_{time.strftime('%Y%m%d_%H%M%S')}"

model = TD3(
    'MlpPolicy',
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=0.0001,
    seed=args.seed,
    tensorboard_log=logdir,
)

# model = TD3.load('models/TD3_32/500.zip', env=env)


TIMESTEPS = args.timesteps
for i in range(args.checkpoints):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=tb_log_name)
    model.save(f"{models_dir}/{TIMESTEPS * (i + 1)}")