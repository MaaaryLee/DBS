import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import TD3

from BGN_MC_Online import BGN_MC_Online

# log things with: tensorboard --logdir=logs/TD3_0 --reload_multifile=True

DEFAULT_H1 = 22
DEFAULT_H2 = 22


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def train_td3(
    *,
    h1: int,
    h2: int,
    timesteps: int,
    checkpoints: int,
    seed: int = 0,
    tb_log_name: Optional[str] = None,
) -> list[Path]:
    models_dir = Path(f"models/TD3_{h1}_{h2}")
    logdir = Path("logs")
    models_dir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[h1, h2], qf=[h1, h2]))

    # Use cached mode for smoke test / offline training path.
    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)

    if not tb_log_name:
        tb_log_name = f"TD3_{h1}_{h2}_{time.strftime('%Y%m%d_%H%M%S')}"

    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0001,
        seed=seed,
        tensorboard_log=str(logdir),
    )

    saved_checkpoints: list[Path] = []
    for index in range(checkpoints):
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=tb_log_name)
        checkpoint_path = models_dir / f"{timesteps * (index + 1)}"
        model.save(str(checkpoint_path))
        saved_checkpoints.append(checkpoint_path.with_suffix(".zip"))

    return saved_checkpoints


def main() -> int:
    args = build_parser().parse_args()
    saved = train_td3(
        h1=args.h1,
        h2=args.h2,
        timesteps=args.timesteps,
        checkpoints=args.checkpoints,
        seed=args.seed,
        tb_log_name=args.tb_log_name,
    )
    print("Saved checkpoints:")
    for checkpoint in saved:
        print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
