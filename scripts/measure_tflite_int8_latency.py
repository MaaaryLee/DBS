import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def _states_path_for_dim(obs_dim: int, default_path: Path) -> Path:
    suffix = default_path.suffix or ".npy"
    if f"_{obs_dim}d" in default_path.stem:
        return default_path.with_suffix(suffix)
    return default_path.with_name(f"{default_path.stem}_{obs_dim}d{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure INT8 TFLite latency.")
    parser.add_argument("--model-path", type=str, default="tflite_actors/model_int8.tflite")
    parser.add_argument("--states-path", type=str, default="states_eval.npy")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inner-repeats", type=int, default=50)
    parser.add_argument("--output-path", type=str, default="results/tflite_int8_latency.json")
    parser.add_argument(
        "--disable-default-delegates",
        action="store_true",
        help="Disable default delegates such as XNNPACK.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"missing {model_path}")

    states_path = Path(args.states_path)
    disable_default_delegates = bool(args.disable_default_delegates)

    interpreter_kwargs = {"model_path": str(model_path), "num_threads": int(args.threads)}
    if disable_default_delegates:
        try:
            interpreter_kwargs["experimental_op_resolver_type"] = (
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
        except Exception:
            pass

    try:
        interpreter = tf.lite.Interpreter(**interpreter_kwargs)
    except TypeError:
        interpreter_kwargs.pop("experimental_op_resolver_type", None)
        interpreter = tf.lite.Interpreter(**interpreter_kwargs)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]
    locked_shape = [1] + list(input_details[0]["shape"][1:])

    if list(input_details[0]["shape"]) != locked_shape:
        interpreter.resize_tensor_input(input_index, locked_shape)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()

    input_scale, input_zero_point = input_details[0]["quantization"]
    if input_scale == 0:
        raise SystemExit("input scale is zero; quantization info missing")

    obs_dim = int(locked_shape[1])
    target_states_path = _states_path_for_dim(obs_dim=obs_dim, default_path=states_path)
    if not target_states_path.exists():
        rng = np.random.default_rng(seed=0)
        states = rng.normal(loc=0.0, scale=1.0, size=(1000, obs_dim)).astype("float32")
        np.save(target_states_path, states)

    states = np.load(target_states_path).astype("float32")
    if states.ndim != 2 or states.shape[1] != obs_dim:
        rng = np.random.default_rng(seed=0)
        states = rng.normal(loc=0.0, scale=1.0, size=(1000, obs_dim)).astype("float32")
        np.save(target_states_path, states)

    sample = states[0:1]
    quantized_sample = np.round(sample / input_scale + input_zero_point).astype(np.int8)
    quantized_sample = quantized_sample.reshape(locked_shape)

    for _ in range(int(args.warmup)):
        for _ in range(int(args.inner_repeats)):
            interpreter.set_tensor(input_index, quantized_sample)
            interpreter.invoke()

    latencies = []
    for _ in range(int(args.runs)):
        start = time.perf_counter()
        for _ in range(int(args.inner_repeats)):
            interpreter.set_tensor(input_index, quantized_sample)
            interpreter.invoke()
        latencies.append((time.perf_counter() - start) * 1000 / int(args.inner_repeats))

    summary = {
        "model": str(model_path),
        "dtype": "int8",
        "obs_dim": int(obs_dim),
        "input_shape": [int(dim) for dim in locked_shape],
        "states_path_used": str(target_states_path),
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "threads": int(args.threads),
        "inner_repeats": int(args.inner_repeats),
        "default_delegates_enabled": not disable_default_delegates,
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
