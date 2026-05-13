"""
Generate strict 6D calibration/held-out splits and replay trajectories.

Why this exists:
- INT8 calibration should see one data split.
- Fidelity should be measured on a different, disjoint split.
- Replay trajectories let us compare FP32 vs INT8 controller behavior over many
  varied held-out sequences instead of relying only on the deterministic cached
  environment smoke test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_states_eval import _compute_cached_6d_state, _make_env  # noqa: E402


def _state_coverage(states: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(v) for v in states.shape],
        "unique_rows": int(len(np.unique(states, axis=0))),
        "per_dim_std": [float(v) for v in np.std(states, axis=0)],
        "per_dim_min": [float(v) for v in np.min(states, axis=0)],
        "per_dim_max": [float(v) for v in np.max(states, axis=0)],
    }


def _compute_sgis_norm(env, sgis_window: np.ndarray) -> float:
    numerator = np.sum(np.average(np.abs(np.fft.fft(sgis_window)), axis=0)[1:20]) - env.sgis_min
    denominator = env.sgis_max - env.sgis_min
    return float(numerator / denominator)


def _extract_window_features(
    env,
    sgis: np.ndarray,
    vgi: np.ndarray,
    vsn: np.ndarray,
    *,
    sgis_start: int,
    signal_start: int,
    sgis_window: int,
    signal_window: int,
    signal_stride: int,
) -> tuple[np.ndarray, float]:
    sgis_slice = sgis[:, sgis_start : sgis_start + sgis_window]
    vgi_slice = vgi[:, signal_start : signal_start + signal_window : signal_stride]
    vsn_slice = vsn[:, signal_start : signal_start + signal_window : signal_stride]
    state = _compute_cached_6d_state(env, sgis_slice, vgi_slice, vsn_slice)
    sgi_norm = _compute_sgis_norm(env, sgis_slice)
    return state.astype(np.float32), sgi_norm


def _sample_unique_pairs(
    *,
    rng: np.random.Generator,
    count: int,
    max_sgis_start: int,
    max_signal_start: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    while len(pairs) < count:
        pair = (
            int(rng.integers(0, max_sgis_start)),
            int(rng.integers(0, max_signal_start)),
        )
        if pair in blocked:
            continue
        blocked.add(pair)
        pairs.append(pair)
    return pairs


def _sample_replay_episode_starts(
    *,
    rng: np.random.Generator,
    episode_count: int,
    episode_steps: int,
    max_sgis_start: int,
    max_signal_start: int,
    sgis_step_delta: int,
    signal_step_delta: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    max_sgis_episode_start = max_sgis_start - (episode_steps - 1) * sgis_step_delta
    max_signal_episode_start = max_signal_start - (episode_steps - 1) * signal_step_delta
    if max_sgis_episode_start <= 0 or max_signal_episode_start <= 0:
        raise ValueError("Replay step deltas are too large for the available cached signals.")

    while len(starts) < episode_count:
        sgis_start = int(rng.integers(0, max_sgis_episode_start))
        signal_start = int(rng.integers(0, max_signal_episode_start))
        pairs = [
            (sgis_start + step * sgis_step_delta, signal_start + step * signal_step_delta)
            for step in range(episode_steps)
        ]
        if any(pair in blocked for pair in pairs):
            continue
        blocked.update(pairs)
        starts.append((sgis_start, signal_start))
    return starts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strict 6D calibration/held-out/replay data.")
    parser.add_argument(
        "--base-states-path",
        type=str,
        default=None,
        help="Optional existing 6D state bank to split directly instead of regenerating from raw cached signals.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calibration-count", type=int, default=1000)
    parser.add_argument("--heldout-count", type=int, default=1000)
    parser.add_argument("--replay-episodes", type=int, default=100)
    parser.add_argument("--replay-steps", type=int, default=10)
    parser.add_argument("--sgis-step-delta", type=int, default=250)
    parser.add_argument("--signal-step-delta", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results/strict_eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_path = output_dir / "states_calibration_6d_strict.npy"
    heldout_path = output_dir / "states_heldout_6d_strict.npy"
    replay_path = output_dir / "replay_episodes_6d_strict.npz"
    calibration_meta_path = output_dir / "states_calibration_6d_strict_metadata.json"
    heldout_meta_path = output_dir / "states_heldout_6d_strict_metadata.json"
    replay_meta_path = output_dir / "replay_episodes_6d_strict_metadata.json"

    if args.base_states_path is not None:
        base_states_path = Path(args.base_states_path)
        states = np.load(base_states_path).astype(np.float32)
        if states.ndim != 2 or states.shape[1] != 6:
            raise SystemExit(f"Expected 2D 6D states at {base_states_path}, got shape={states.shape}")

        replay_total = int(args.replay_episodes) * int(args.replay_steps)
        required = int(args.calibration_count) + int(args.heldout_count)
        if states.shape[0] < required:
            raise SystemExit(
                f"Not enough rows in {base_states_path}: need at least {required}, found {states.shape[0]}"
            )
        if int(args.heldout_count) < replay_total:
            raise SystemExit(
                f"heldout-count must be at least replay_episodes * replay_steps ({replay_total})"
            )

        rng = np.random.default_rng(seed=args.seed)
        permutation = rng.permutation(states.shape[0])
        calibration_indices = permutation[: int(args.calibration_count)]
        heldout_indices = permutation[
            int(args.calibration_count) : int(args.calibration_count) + int(args.heldout_count)
        ]

        calibration_states = states[calibration_indices].astype(np.float32)
        heldout_states = states[heldout_indices].astype(np.float32)
        replay_observations = heldout_states[:replay_total].reshape(
            int(args.replay_episodes), int(args.replay_steps), 6
        )

        np.save(calibration_path, calibration_states)
        np.save(heldout_path, heldout_states)
        np.savez_compressed(replay_path, observations=replay_observations)

        base_metadata = {
            "generator": str(Path(__file__).resolve()),
            "seed": int(args.seed),
            "source": "existing_state_bank_split",
            "base_states_path": str(base_states_path.resolve()),
        }

        calibration_meta = {
            **base_metadata,
            "artifact": str(calibration_path.resolve()),
            "coverage": _state_coverage(calibration_states),
            "source_indices": [int(v) for v in calibration_indices],
        }
        heldout_meta = {
            **base_metadata,
            "artifact": str(heldout_path.resolve()),
            "coverage": _state_coverage(heldout_states),
            "source_indices": [int(v) for v in heldout_indices],
        }
        replay_meta = {
            **base_metadata,
            "artifact": str(replay_path.resolve()),
            "episodes": int(args.replay_episodes),
            "steps_per_episode": int(args.replay_steps),
            "observation_coverage": _state_coverage(replay_observations.reshape(-1, 6)),
            "replay_source_indices": [int(v) for v in heldout_indices[:replay_total]],
        }

        calibration_meta_path.write_text(json.dumps(calibration_meta, indent=2))
        heldout_meta_path.write_text(json.dumps(heldout_meta, indent=2))
        replay_meta_path.write_text(json.dumps(replay_meta, indent=2))

        print(
            json.dumps(
                {
                    "mode": "existing_state_bank_split",
                    "calibration_states": str(calibration_path.resolve()),
                    "heldout_states": str(heldout_path.resolve()),
                    "replay_episodes": str(replay_path.resolve()),
                    "calibration_unique_rows": calibration_meta["coverage"]["unique_rows"],
                    "heldout_unique_rows": heldout_meta["coverage"]["unique_rows"],
                    "replay_unique_rows": replay_meta["observation_coverage"]["unique_rows"],
                },
                indent=2,
            )
        )
        return 0

    env = _make_env(source="online_cached", tmax=1100, pd=True, mode=None)
    if not getattr(env, "has_precomputed_data", False):
        raise SystemExit("Cached strict 6D generation requires precomputed MATLAB data.")

    sgis = np.asarray(env.sgis, dtype=np.float64)
    vgi = np.asarray(env.vgi, dtype=np.float64)
    vsn = np.asarray(env.vsn, dtype=np.float64)

    sgis_window = min(2000, sgis.shape[1])
    signal_window = min(10000, vgi.shape[1])
    signal_stride = 2
    max_sgis_start = max(1, sgis.shape[1] - sgis_window + 1)
    max_signal_start = max(1, vgi.shape[1] - signal_window + 1)

    rng = np.random.default_rng(seed=args.seed)
    blocked_pairs: set[tuple[int, int]] = set()

    calibration_pairs = _sample_unique_pairs(
        rng=rng,
        count=int(args.calibration_count),
        max_sgis_start=max_sgis_start,
        max_signal_start=max_signal_start,
        blocked=blocked_pairs,
    )
    heldout_pairs = _sample_unique_pairs(
        rng=rng,
        count=int(args.heldout_count),
        max_sgis_start=max_sgis_start,
        max_signal_start=max_signal_start,
        blocked=blocked_pairs,
    )
    replay_starts = _sample_replay_episode_starts(
        rng=rng,
        episode_count=int(args.replay_episodes),
        episode_steps=int(args.replay_steps),
        max_sgis_start=max_sgis_start,
        max_signal_start=max_signal_start,
        sgis_step_delta=int(args.sgis_step_delta),
        signal_step_delta=int(args.signal_step_delta),
        blocked=blocked_pairs,
    )

    calibration_states = []
    calibration_sgi = []
    for sgis_start, signal_start in calibration_pairs:
        state, sgi_norm = _extract_window_features(
            env,
            sgis,
            vgi,
            vsn,
            sgis_start=sgis_start,
            signal_start=signal_start,
            sgis_window=sgis_window,
            signal_window=signal_window,
            signal_stride=signal_stride,
        )
        calibration_states.append(state)
        calibration_sgi.append(sgi_norm)

    heldout_states = []
    heldout_sgi = []
    for sgis_start, signal_start in heldout_pairs:
        state, sgi_norm = _extract_window_features(
            env,
            sgis,
            vgi,
            vsn,
            sgis_start=sgis_start,
            signal_start=signal_start,
            sgis_window=sgis_window,
            signal_window=signal_window,
            signal_stride=signal_stride,
        )
        heldout_states.append(state)
        heldout_sgi.append(sgi_norm)

    replay_observations = np.zeros((args.replay_episodes, args.replay_steps, 6), dtype=np.float32)
    replay_sgi_norm = np.zeros((args.replay_episodes, args.replay_steps), dtype=np.float32)
    replay_sgis_starts = np.zeros((args.replay_episodes, args.replay_steps), dtype=np.int32)
    replay_signal_starts = np.zeros((args.replay_episodes, args.replay_steps), dtype=np.int32)

    for episode_index, (sgis_start, signal_start) in enumerate(replay_starts):
        for step in range(args.replay_steps):
            episode_sgis_start = sgis_start + step * args.sgis_step_delta
            episode_signal_start = signal_start + step * args.signal_step_delta
            state, sgi_norm = _extract_window_features(
                env,
                sgis,
                vgi,
                vsn,
                sgis_start=episode_sgis_start,
                signal_start=episode_signal_start,
                sgis_window=sgis_window,
                signal_window=signal_window,
                signal_stride=signal_stride,
            )
            replay_observations[episode_index, step] = state
            replay_sgi_norm[episode_index, step] = sgi_norm
            replay_sgis_starts[episode_index, step] = episode_sgis_start
            replay_signal_starts[episode_index, step] = episode_signal_start

    calibration_states = np.stack(calibration_states, axis=0).astype(np.float32)
    heldout_states = np.stack(heldout_states, axis=0).astype(np.float32)
    calibration_sgi = np.asarray(calibration_sgi, dtype=np.float32)
    heldout_sgi = np.asarray(heldout_sgi, dtype=np.float32)

    np.save(calibration_path, calibration_states)
    np.save(heldout_path, heldout_states)
    np.savez_compressed(
        replay_path,
        observations=replay_observations,
        sgi_norm=replay_sgi_norm,
        sgis_start=replay_sgis_starts,
        signal_start=replay_signal_starts,
    )

    base_metadata = {
        "generator": str(Path(__file__).resolve()),
        "seed": int(args.seed),
        "source": "online_cached_precomputed_windows",
        "sgis_window": int(sgis_window),
        "signal_window": int(signal_window),
        "signal_stride": int(signal_stride),
        "max_sgis_start": int(max_sgis_start),
        "max_signal_start": int(max_signal_start),
    }

    calibration_meta = {
        **base_metadata,
        "artifact": str(calibration_path.resolve()),
        "sgi_norm_summary": {
            "mean": float(np.mean(calibration_sgi)),
            "std": float(np.std(calibration_sgi)),
            "min": float(np.min(calibration_sgi)),
            "max": float(np.max(calibration_sgi)),
        },
        "coverage": _state_coverage(calibration_states),
        "window_pairs": [[int(a), int(b)] for a, b in calibration_pairs],
    }
    heldout_meta = {
        **base_metadata,
        "artifact": str(heldout_path.resolve()),
        "sgi_norm_summary": {
            "mean": float(np.mean(heldout_sgi)),
            "std": float(np.std(heldout_sgi)),
            "min": float(np.min(heldout_sgi)),
            "max": float(np.max(heldout_sgi)),
        },
        "coverage": _state_coverage(heldout_states),
        "window_pairs": [[int(a), int(b)] for a, b in heldout_pairs],
    }
    replay_meta = {
        **base_metadata,
        "artifact": str(replay_path.resolve()),
        "episodes": int(args.replay_episodes),
        "steps_per_episode": int(args.replay_steps),
        "sgis_step_delta": int(args.sgis_step_delta),
        "signal_step_delta": int(args.signal_step_delta),
        "observation_coverage": _state_coverage(replay_observations.reshape(-1, 6)),
        "sgi_norm_summary": {
            "mean": float(np.mean(replay_sgi_norm)),
            "std": float(np.std(replay_sgi_norm)),
            "min": float(np.min(replay_sgi_norm)),
            "max": float(np.max(replay_sgi_norm)),
        },
        "episode_starts": [[int(a), int(b)] for a, b in replay_starts],
    }

    calibration_meta_path.write_text(json.dumps(calibration_meta, indent=2))
    heldout_meta_path.write_text(json.dumps(heldout_meta, indent=2))
    replay_meta_path.write_text(json.dumps(replay_meta, indent=2))

    print(
        json.dumps(
            {
                "calibration_states": str(calibration_path.resolve()),
                "heldout_states": str(heldout_path.resolve()),
                "replay_episodes": str(replay_path.resolve()),
                "calibration_unique_rows": calibration_meta["coverage"]["unique_rows"],
                "heldout_unique_rows": heldout_meta["coverage"]["unique_rows"],
                "replay_unique_rows": replay_meta["observation_coverage"]["unique_rows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
