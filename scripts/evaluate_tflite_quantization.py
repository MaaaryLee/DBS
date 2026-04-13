"""
Evaluate TFLite FP32 vs INT8 models on fidelity and control behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_interpreter(model_path: Path, threads: int = 1):
    import tensorflow as tf  # type: ignore

    return tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)


def _run_interpreter(interpreter, observations: np.ndarray) -> dict[str, Any]:
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    outputs = []
    sat_min = 0
    sat_max = 0
    total_values = int(np.prod(observations.shape))

    for obs in observations.astype(np.float32):
        if input_details["dtype"] == np.float32:
            tensor = obs.reshape(1, -1).astype(np.float32)
        elif input_details["dtype"] == np.int8:
            scale, zero_point = input_details["quantization"]
            q = np.round(obs / scale + zero_point)
            sat_min += int(np.sum(q < -128))
            sat_max += int(np.sum(q > 127))
            q = np.clip(q, -128, 127).astype(np.int8)
            tensor = q.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported input dtype: {input_details['dtype']}")

        interpreter.set_tensor(input_details["index"], tensor)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])

        if output_details["dtype"] == np.float32:
            out = raw_output.astype(np.float32)
        elif output_details["dtype"] == np.int8:
            scale, zero_point = output_details["quantization"]
            out = (raw_output.astype(np.float32) - zero_point) * scale
        else:
            raise ValueError(f"Unsupported output dtype: {output_details['dtype']}")

        outputs.append(out[0].astype(np.float32))

    sat_total = sat_min + sat_max
    return {
        "outputs": np.stack(outputs, axis=0),
        "input_details": {
            "shape": [int(v) for v in input_details["shape"]],
            "dtype": str(input_details["dtype"]),
            "quantization": [float(input_details["quantization"][0]), int(input_details["quantization"][1])],
        },
        "output_details": {
            "shape": [int(v) for v in output_details["shape"]],
            "dtype": str(output_details["dtype"]),
            "quantization": [float(output_details["quantization"][0]), int(output_details["quantization"][1])],
        },
        "input_saturation": {
            "sat_min_count": sat_min,
            "sat_max_count": sat_max,
            "sat_total_count": sat_total,
            "sat_total_fraction": (sat_total / total_values) if total_values else 0.0,
        },
    }


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _state_coverage(states: np.ndarray) -> dict[str, Any]:
    per_dim_std = np.std(states, axis=0)
    per_dim_min = np.min(states, axis=0)
    per_dim_max = np.max(states, axis=0)
    unique_rows = int(len(np.unique(states, axis=0)))
    return {
        "unique_rows": unique_rows,
        "per_dim_std": [float(v) for v in per_dim_std],
        "per_dim_min": [float(v) for v in per_dim_min],
        "per_dim_max": [float(v) for v in per_dim_max],
        "degenerate": bool(unique_rows <= 1 or (per_dim_std.size and float(np.max(per_dim_std)) < 1e-6)),
    }


def _output_coverage(outputs: np.ndarray) -> dict[str, Any]:
    return {
        "mean": [float(v) for v in np.mean(outputs, axis=0)],
        "std": [float(v) for v in np.std(outputs, axis=0)],
        "min": [float(v) for v in np.min(outputs, axis=0)],
        "max": [float(v) for v in np.max(outputs, axis=0)],
    }


def _fidelity_metrics(fp32_outputs: np.ndarray, int8_outputs: np.ndarray) -> dict[str, Any]:
    abs_diff = np.abs(fp32_outputs - int8_outputs)
    mse = np.mean((fp32_outputs - int8_outputs) ** 2)
    mae = np.mean(abs_diff)

    return {
        "mse": float(mse),
        "mae": float(mae),
        "max_abs_diff": float(abs_diff.max()),
        "p50_abs_diff": float(np.percentile(abs_diff, 50)),
        "p90_abs_diff": float(np.percentile(abs_diff, 90)),
        "per_output": [
            {
                "index": int(i),
                "mae": float(np.mean(abs_diff[:, i])),
                "max_abs_diff": float(np.max(abs_diff[:, i])),
                "pearson_r": _corrcoef(fp32_outputs[:, i], int8_outputs[:, i]),
            }
            for i in range(fp32_outputs.shape[1])
        ],
    }


def _predict_action(interpreter, observation: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details["dtype"] == np.float32:
        tensor = observation.reshape(1, -1).astype(np.float32)
    else:
        scale, zero_point = input_details["quantization"]
        q = np.round(observation / scale + zero_point)
        q = np.clip(q, -128, 127).astype(np.int8)
        tensor = q.reshape(1, -1)

    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details["index"])

    if output_details["dtype"] == np.float32:
        return raw_output[0].astype(np.float32)
    scale, zero_point = output_details["quantization"]
    return ((raw_output.astype(np.float32) - zero_point) * scale)[0]


def _action_to_stim(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    frequency_hz = 185.0 * ((action[..., 0] + 1.0) / 2.0)
    amplitude_ma = 5000.0 * ((action[..., 1] + 1.0) / 2.0)
    return frequency_hz.astype(np.float32), amplitude_ma.astype(np.float32)


def _offline_reward_from_sgi_and_action(sgi_norm: np.ndarray, action: np.ndarray) -> np.ndarray:
    epsilon = 0.68
    frequency_hz, amplitude_ma = _action_to_stim(action)
    r2 = 0.85 * (frequency_hz / 185.0) + 0.15 * (amplitude_ma / 5000.0)
    return (epsilon * -np.asarray(sgi_norm, dtype=np.float32) + (1.0 - epsilon) * -r2).astype(np.float32)


def _evaluate_replay(model_path: Path, replay_path: Path) -> dict[str, Any]:
    payload = np.load(replay_path)
    observations = np.asarray(payload["observations"], dtype=np.float32)
    sgi_norm = np.asarray(payload["sgi_norm"], dtype=np.float32) if "sgi_norm" in payload else None
    episodes, steps, obs_dim = observations.shape

    interpreter = _make_interpreter(model_path=model_path, threads=1)
    interpreter.allocate_tensors()

    actions = np.zeros((episodes, steps, 2), dtype=np.float32)
    for episode in range(episodes):
        for step in range(steps):
            actions[episode, step] = _predict_action(interpreter, observations[episode, step])

    frequency_hz, amplitude_ma = _action_to_stim(actions)

    result = {
        "replay_path": str(replay_path),
        "episodes": int(episodes),
        "steps_per_episode": int(steps),
        "observation_dim": int(obs_dim),
        "action_summary": {
            "mean": [float(v) for v in np.mean(actions.reshape(-1, 2), axis=0)],
            "std": [float(v) for v in np.std(actions.reshape(-1, 2), axis=0)],
            "min": [float(v) for v in np.min(actions.reshape(-1, 2), axis=0)],
            "max": [float(v) for v in np.max(actions.reshape(-1, 2), axis=0)],
        },
        "stim_summary": {
            "frequency_hz_mean": float(np.mean(frequency_hz)),
            "frequency_hz_std": float(np.std(frequency_hz)),
            "frequency_hz_min": float(np.min(frequency_hz)),
            "frequency_hz_max": float(np.max(frequency_hz)),
            "amplitude_ma_mean": float(np.mean(amplitude_ma)),
            "amplitude_ma_std": float(np.std(amplitude_ma)),
            "amplitude_ma_min": float(np.min(amplitude_ma)),
            "amplitude_ma_max": float(np.max(amplitude_ma)),
        },
        "actions": actions,
    }
    if sgi_norm is not None:
        episode_returns = np.sum(_offline_reward_from_sgi_and_action(sgi_norm, actions), axis=1)
        result["surrogate_return"] = {
            "mean": float(np.mean(episode_returns)),
            "std": float(np.std(episode_returns, ddof=0)),
            "min": float(np.min(episode_returns)),
            "max": float(np.max(episode_returns)),
            "episode_trace": [float(v) for v in episode_returns],
        }
    return result


def _evaluate_env(model_path: Path, episodes: int, seed: int) -> dict[str, Any]:
    from BGN_MC_Online import BGN_MC_Online  # type: ignore

    interpreter = _make_interpreter(model_path=model_path, threads=1)
    interpreter.allocate_tensors()

    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)
    rewards = []
    lengths = []
    freqs = []
    amps = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = _predict_action(interpreter, np.asarray(observation, dtype=np.float32))
            freqs.append(float(185 * ((action[0] + 1.0) / 2.0)))
            amps.append(float(5000 * ((action[1] + 1.0) / 2.0)))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=0)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(lengths)),
        "mean_frequency_hz": float(np.mean(freqs)) if freqs else math.nan,
        "mean_amplitude_ma": float(np.mean(amps)) if amps else math.nan,
        "reward_trace": [float(v) for v in rewards],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TFLite FP32 vs INT8 quantization fidelity.")
    parser.add_argument("--fp32-model", required=True)
    parser.add_argument("--int8-model", required=True)
    parser.add_argument("--states-path", default="states_eval_6d.npy")
    parser.add_argument("--replay-path", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    fp32_model = Path(args.fp32_model)
    int8_model = Path(args.int8_model)
    states = np.load(args.states_path).astype(np.float32)

    fp32_eval = _run_interpreter(_make_interpreter(fp32_model), states)
    int8_eval = _run_interpreter(_make_interpreter(int8_model), states)
    state_coverage = _state_coverage(states)
    fp32_output_coverage = _output_coverage(fp32_eval["outputs"])
    int8_output_coverage = _output_coverage(int8_eval["outputs"])

    replay_results = None
    if args.replay_path:
        replay_path = Path(args.replay_path)
        fp32_replay = _evaluate_replay(fp32_model, replay_path)
        int8_replay = _evaluate_replay(int8_model, replay_path)
        fp32_actions = fp32_replay.pop("actions")
        int8_actions = int8_replay.pop("actions")
        action_diff = np.abs(fp32_actions - int8_actions)
        fp32_freq, fp32_amp = _action_to_stim(fp32_actions)
        int8_freq, int8_amp = _action_to_stim(int8_actions)
        agreement = {
            "action_mae": float(np.mean(action_diff)),
            "action_max_abs_diff": float(np.max(action_diff)),
            "action_p90_abs_diff": float(np.percentile(action_diff, 90)),
            "frequency_hz_mae": float(np.mean(np.abs(fp32_freq - int8_freq))),
            "amplitude_ma_mae": float(np.mean(np.abs(fp32_amp - int8_amp))),
            "per_action_dim": [
                {
                    "index": int(i),
                    "mae": float(np.mean(action_diff[..., i])),
                    "max_abs_diff": float(np.max(action_diff[..., i])),
                    "pearson_r": _corrcoef(fp32_actions[..., i].reshape(-1), int8_actions[..., i].reshape(-1)),
                }
                for i in range(fp32_actions.shape[-1])
            ],
        }
        if "surrogate_return" in fp32_replay and "surrogate_return" in int8_replay:
            fp32_returns = np.asarray(fp32_replay["surrogate_return"]["episode_trace"], dtype=np.float32)
            int8_returns = np.asarray(int8_replay["surrogate_return"]["episode_trace"], dtype=np.float32)
            agreement.update(
                {
                    "episode_return_mae": float(np.mean(np.abs(fp32_returns - int8_returns))),
                    "episode_return_max_abs_diff": float(np.max(np.abs(fp32_returns - int8_returns))),
                    "episode_return_pearson_r": _corrcoef(fp32_returns, int8_returns),
                }
            )
        replay_results = {
            "fp32": fp32_replay,
            "int8": int8_replay,
            "agreement": agreement,
            "interpretation": (
                "Replay control agreement uses many varied held-out trajectories and compares "
                "the FP32 and INT8 action sequences on the same inputs. It is stronger than the "
                "deterministic cached smoke test, but it is still not a full fresh closed-loop "
                "simulation benchmark."
            ),
        }

    results = {
        "fp32_model": str(fp32_model),
        "int8_model": str(int8_model),
        "states_path": str(args.states_path),
        "states_shape": [int(v) for v in states.shape],
        "state_coverage": state_coverage,
        "model_size_bytes": {
            "fp32": fp32_model.stat().st_size,
            "int8": int8_model.stat().st_size,
            "reduction_percent": (1.0 - int8_model.stat().st_size / fp32_model.stat().st_size) * 100.0,
        },
        "fp32_io": {
            "input": fp32_eval["input_details"],
            "output": fp32_eval["output_details"],
        },
        "int8_io": {
            "input": int8_eval["input_details"],
            "output": int8_eval["output_details"],
        },
        "input_saturation": int8_eval["input_saturation"],
        "output_coverage": {
            "fp32": fp32_output_coverage,
            "int8": int8_output_coverage,
        },
        "fidelity": _fidelity_metrics(fp32_eval["outputs"], int8_eval["outputs"]),
        "env_eval": {
            "fp32": _evaluate_env(fp32_model, episodes=args.episodes, seed=args.seed),
            "int8": _evaluate_env(int8_model, episodes=args.episodes, seed=args.seed),
        },
    }
    if replay_results is not None:
        results["replay_eval"] = replay_results

    warnings = []
    if state_coverage["degenerate"]:
        warnings.append("State coverage is degenerate; fidelity metrics are only valid at one repeated input point.")
    if any(math.isnan(v["pearson_r"]) for v in results["fidelity"]["per_output"]):
        warnings.append("At least one per-output Pearson correlation is NaN because the evaluated outputs have near-zero variance.")
    if (
        abs(results["env_eval"]["fp32"]["std_reward"]) < 1e-9
        and abs(results["env_eval"]["int8"]["std_reward"]) < 1e-9
    ):
        warnings.append("Cached-environment reward traces are deterministic in this setup; treat env_eval as a functional smoke test, not a stochastic control benchmark.")
    if replay_results is not None:
        warnings.append(
            "Replay_eval is stronger than the cached smoke test because it covers many varied held-out trajectories, "
            "but it still replays cached signals rather than running a fresh closed-loop MATLAB simulation."
        )
    results["warnings"] = warnings

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
