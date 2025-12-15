"""
Generate a realistic `states_eval.npy` by rolling the DBS environment.

Why:
- `states_eval.npy` is used for INT8 calibration + fair latency/fidelity benchmarking.
- Synthetic Gaussian states are convenient but not realistic.

Default behavior uses `BGN_MC_Online` with cached data (no MATLAB engine required).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


# Ensure repo root is on sys.path so `BGN_MC_Online.py` / `BGN_MC.py` shims are importable
# when running this script as `python scripts/generate_states_eval.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _make_env(source: str, tmax: int, pd: bool, mode: str | None):
    if source == "online_cached":
        from BGN_MC_Online import BGN_MC_Online

        if mode is not None:
            raise ValueError("online_cached source does not support mode selection (always 6D).")
        return BGN_MC_Online(tmax=tmax, pd=pd, use_matlab_online=False)
    if source == "matlab_engine":
        from BGN_MC import BGN_MC

        if mode is None:
            return BGN_MC(tmax=tmax, pd=pd)
        return BGN_MC(tmax=tmax, pd=pd, mode=mode)
    raise ValueError(f"Unknown source: {source}")


def collect_states(
    *,
    num_states: int,
    source: str,
    tmax: int,
    pd: bool,
    steps_per_episode: int,
    seed: int,
    mode: str | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    env = _make_env(source=source, tmax=tmax, pd=pd, mode=mode)

    states = []
    obs, _ = env.reset(seed=seed)
    states.append(np.asarray(obs, dtype=np.float32))
    terminated = False
    truncated = False
    steps_left = steps_per_episode

    while len(states) < num_states:
        if terminated or truncated or steps_left <= 0:
            obs, _ = env.reset()
            terminated = False
            truncated = False
            steps_left = steps_per_episode

        # Prefer environment sampling; fall back to deterministic RNG if needed.
        try:
            action = env.action_space.sample()
        except Exception:
            low = np.asarray(env.action_space.low, dtype=np.float32)
            high = np.asarray(env.action_space.high, dtype=np.float32)
            action = rng.uniform(low=low, high=high).astype(np.float32)

        obs, _, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(obs, dtype=np.float32))
        steps_left -= 1
        
        # Progress indicator (print every 10% or every 50 states, whichever is more frequent)
        if len(states) % max(1, num_states // 20) == 0 or len(states) % 50 == 0:
            print(f"[Progress] Collected {len(states)}/{num_states} states ({100*len(states)//num_states}%)", flush=True)

    arr = np.stack(states[:num_states], axis=0).astype(np.float32)
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate realistic states_eval.npy by rolling the DBS environment."
    )
    parser.add_argument(
        "--source",
        choices=("online_cached", "matlab_engine"),
        default="online_cached",
        help="Where to collect states from (default: online_cached).",
    )
    parser.add_argument(
        "--mode",
        choices=("hvgi", "hsgi", "hvgi_sgi"),
        default=None,
        help="Observation mode (MATLAB engine only). If omitted, uses env default.",
    )
    parser.add_argument("--num-states", type=int, default=1000, help="Number of states to save.")
    parser.add_argument("--tmax", type=int, default=1100, help="Environment tmax.")
    parser.add_argument("--pd", action="store_true", help="Use Parkinsonian mode (pd=True).")
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=10,
        help="How many random steps to take before resetting.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        default="states_eval.npy",
        help="Output .npy path (default: states_eval.npy).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    states = collect_states(
        num_states=args.num_states,
        source=args.source,
        tmax=args.tmax,
        pd=bool(args.pd),
        steps_per_episode=args.steps_per_episode,
        seed=args.seed,
        mode=args.mode,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, states)
    print(f"[OK] Wrote {out_path} with shape={states.shape} dtype={states.dtype} source={args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


