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

import antropy
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


def _compute_cached_6d_state(env, sgis_window: np.ndarray, vgi_window: np.ndarray, vsn_window: np.ndarray) -> np.ndarray:
    eps = 1e-12

    sd = (np.average(np.std(sgis_window, axis=1)) - env.sd_min) / (env.sd_max - env.sd_min)

    activity = float(np.average(np.var(sgis_window, axis=1)))
    activity = max(activity, eps)
    activity_norm = (activity - env.A_min) / (env.A_max - env.A_min)

    mobility = float(np.average(np.sqrt(np.maximum(np.var(np.diff(sgis_window, axis=1), axis=1), 0.0) / activity)))
    mobility_norm = (mobility - env.M_min) / (env.M_max - env.M_min)

    complexity = float(
        np.average(
            np.sqrt(np.maximum(np.var(np.diff(np.diff(sgis_window, axis=1), axis=1), axis=1), 0.0) / activity)
            / max(mobility, eps)
        )
    )
    complexity_norm = (complexity - env.C_min) / (env.C_max - env.C_min)

    pb = (
        np.sum(np.average(np.abs(np.fft.fft(vgi_window)) / 0.1, axis=0)[12:31]) - env.Pb_min
    ) / (env.Pb_max - env.Pb_min)

    sampen_values = []
    for row in vsn_window:
        try:
            sampen_values.append(float(antropy.sample_entropy(row)))
        except Exception:
            continue
    if sampen_values:
        sampen = (float(np.mean(sampen_values)) - env.SampEn_min) / (env.SampEn_max - env.SampEn_min)
    else:
        sampen = 0.5

    return np.asarray((sd, activity_norm, mobility_norm, complexity_norm, pb, sampen), dtype=np.float32)


def _collect_windowed_cached_states(env, *, num_states: int, seed: int) -> np.ndarray:
    """
    The cached 6D environment currently reuses one precomputed signal bundle, so a
    simple env rollout produces the same observation repeatedly. To recover a usable
    calibration/evaluation set, derive feature vectors from random time windows of
    the cached MATLAB signals instead.
    """

    if not getattr(env, "has_precomputed_data", False):
        raise RuntimeError("Cached 6D state generation requires precomputed MATLAB data")

    rng = np.random.default_rng(seed=seed)
    sgis = np.asarray(env.sgis, dtype=np.float64)
    vgi = np.asarray(env.vgi, dtype=np.float64)
    vsn = np.asarray(env.vsn, dtype=np.float64)

    sgis_window = min(2000, sgis.shape[1])
    signal_window = min(10000, vgi.shape[1])
    signal_stride = 2

    max_sgis_start = max(1, sgis.shape[1] - sgis_window + 1)
    max_signal_start = max(1, vgi.shape[1] - signal_window + 1)

    states = []
    for _ in range(num_states):
        sgis_start = int(rng.integers(0, max_sgis_start))
        signal_start = int(rng.integers(0, max_signal_start))

        sgis_slice = sgis[:, sgis_start : sgis_start + sgis_window]
        vgi_slice = vgi[:, signal_start : signal_start + signal_window : signal_stride]
        vsn_slice = vsn[:, signal_start : signal_start + signal_window : signal_stride]
        states.append(_compute_cached_6d_state(env, sgis_slice, vgi_slice, vsn_slice))

        if len(states) % max(1, num_states // 20) == 0 or len(states) % 50 == 0:
            print(f"[Progress] Collected {len(states)}/{num_states} states ({100 * len(states) // num_states}%)", flush=True)

    return np.stack(states, axis=0).astype(np.float32)


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

    if source == "online_cached":
        return _collect_windowed_cached_states(env, num_states=num_states, seed=seed)

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

