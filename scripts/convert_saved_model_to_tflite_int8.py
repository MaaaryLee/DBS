import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf


def _states_path_for_dim(obs_dim: int, default_path: Path) -> Path:
    suffix = default_path.suffix or ".npy"
    if f"_{obs_dim}d" in default_path.stem:
        return default_path.with_suffix(suffix)
    return default_path.with_name(f"{default_path.stem}_{obs_dim}d{suffix}")


def _state_coverage(states: np.ndarray) -> dict[str, object]:
    per_dim_std = np.std(states, axis=0)
    return {
        "unique_rows": int(len(np.unique(states, axis=0))),
        "per_dim_std": [float(v) for v in per_dim_std],
        "max_std": float(np.max(per_dim_std)) if per_dim_std.size else 0.0,
    }


def _is_degenerate_states(states: np.ndarray) -> bool:
    coverage = _state_coverage(states)
    return bool(coverage["unique_rows"] <= 1 or coverage["max_std"] < 1e-6)


def _collect_real_states(obs_dim: int, path: Path) -> Optional[np.ndarray]:
    """
    Prefer realistic cached DBS states for 6D models so PTQ calibration reflects the
    training distribution. Fall back to synthetic data only if collection is unavailable.
    """

    if obs_dim != 6:
        return None

    try:
        from generate_states_eval import collect_states

        states = collect_states(
            num_states=1000,
            source="online_cached",
            tmax=1100,
            pd=True,
            steps_per_episode=10,
            seed=0,
            mode=None,
        ).astype("float32")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, states)
        print(
            f"[OK] Collected realistic {path.name} with shape={states.shape} "
            f"for INT8 calibration; coverage={_state_coverage(states)}"
        )
        return states
    except Exception as exc:
        print(f"[WARNING] Could not collect realistic states for INT8 calibration: {exc}")
        return None


def _ensure_states(saved_model_dir: str, path: Path) -> np.ndarray:
    loaded = tf.saved_model.load(saved_model_dir)
    func = loaded.signatures.get("serving_default")
    if func is None:
        raise SystemExit("SavedModel has no 'serving_default' signature to infer input shape.")

    input_tensor = None
    for _, tensor in func.structured_input_signature[1].items():
        input_tensor = tensor
        break
    if input_tensor is None:
        raise SystemExit("Could not infer input tensor from SavedModel signature.")

    shape = input_tensor.shape.as_list()
    if len(shape) != 2:
        raise SystemExit(f"Expected 2D input [batch, features], got shape={shape}")
    obs_dim = int(shape[1])
    target_path = _states_path_for_dim(obs_dim=obs_dim, default_path=path)

    rng = np.random.default_rng(seed=0)
    states = rng.normal(loc=0.0, scale=1.0, size=(1000, obs_dim)).astype("float32")

    for candidate in (target_path, path):
        if not candidate.exists():
            continue
        existing = np.load(candidate).astype("float32")
        if existing.ndim == 2 and existing.shape[1] == obs_dim:
            if obs_dim == 6 and _is_degenerate_states(existing):
                print(
                    f"[WARNING] Ignoring degenerate {candidate.name} for INT8 calibration; "
                    f"coverage={_state_coverage(existing)}"
                )
                continue
            return existing

    real_states = _collect_real_states(obs_dim=obs_dim, path=target_path)
    if real_states is not None:
        return real_states

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, states)
    print(f"[OK] Synthesized {target_path.name} with shape={states.shape} for INT8 calibration")
    return states


def _load_static_wrapper(saved_model_dir: str, batch_size: int = 1):
    loaded = tf.saved_model.load(saved_model_dir)
    serving = loaded.signatures.get("serving_default")
    if serving is None:
        raise SystemExit("SavedModel has no 'serving_default' signature.")

    input_items = list(serving.structured_input_signature[1].items())
    if len(input_items) != 1:
        raise SystemExit("Expected exactly one input tensor in serving_default.")

    input_name, input_tensor = input_items[0]
    shape = input_tensor.shape.as_list()
    if len(shape) != 2:
        raise SystemExit(f"Expected rank-2 input [batch, features], got shape={shape}")
    obs_dim = int(shape[1])

    class StaticWrapper(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec([batch_size, obs_dim], input_tensor.dtype, name=input_name)
            ]
        )
        def serve(self, observation):
            return serving(**{input_name: observation})

    wrapper = StaticWrapper()
    concrete = wrapper.serve.get_concrete_function()
    return wrapper, concrete, obs_dim


def convert_saved_model_to_tflite_int8(
    saved_model_dir: str = "tf_model",
    output_path: Path = Path("tflite_actors/model_int8.tflite"),
    states_path: Path = Path("states_eval.npy"),
) -> bool:
    wrapper, concrete, obs_dim = _load_static_wrapper(saved_model_dir, batch_size=1)
    states = _ensure_states(saved_model_dir, states_path)

    def representative_dataset():
        for row in states[:1000]:
            yield [row.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], wrapper)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_model = converter.convert()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)
        print(f"[OK] Saved {output_path} with static input shape [1, {obs_dim}]")
        return True
    except Exception as exc:
        import traceback

        print(f"[X] Conversion failed: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a SavedModel to a fully integer TFLite model.")
    parser.add_argument("--saved-model-dir", type=str, default="tf_model")
    parser.add_argument("--output-path", type=str, default="tflite_actors/model_int8.tflite")
    parser.add_argument("--states-path", type=str, default="states_eval.npy")
    args = parser.parse_args()

    success = convert_saved_model_to_tflite_int8(
        saved_model_dir=args.saved_model_dir,
        output_path=Path(args.output_path),
        states_path=Path(args.states_path),
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
