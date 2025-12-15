"""
Inspect a TFLite model for:
- input/output shapes + dtypes
- quantization params (scale/zero_point)
- graph operator breakdown (via tf.lite.experimental.Analyzer)

Useful to verify whether an "INT8" model is actually fully-integer and whether
dynamic shapes are present (which can block full delegation/optimization).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _make_interpreter(tf, model_path: Path, threads: int):
    disable_default_delegates = os.environ.get("TFLITE_DISABLE_DEFAULT_DELEGATES", "0") == "1"
    kwargs = {"model_path": str(model_path), "num_threads": int(threads)}
    if disable_default_delegates:
        try:
            kwargs["experimental_op_resolver_type"] = (
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            )
        except Exception:
            pass
    try:
        return tf.lite.Interpreter(**kwargs)
    except TypeError:
        kwargs.pop("experimental_op_resolver_type", None)
        return tf.lite.Interpreter(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a TFLite model.")
    parser.add_argument("--model", type=str, required=True, help="Path to .tflite model")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--analyze", action="store_true", help="Print tf.lite.experimental.Analyzer output")
    args = parser.parse_args()

    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise SystemExit("TensorFlow is required to inspect TFLite models.") from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"missing {model_path}")

    print("=" * 70)
    print(f"MODEL: {model_path}")
    print(f"TFLITE_DISABLE_DEFAULT_DELEGATES={os.environ.get('TFLITE_DISABLE_DEFAULT_DELEGATES', '(unset)')}")
    print("=" * 70)

    interpreter = _make_interpreter(tf, model_path=model_path, threads=args.threads)
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    print("\nInputs:")
    for i, d in enumerate(inputs):
        print(
            f"  [{i}] name={d.get('name')} shape={d.get('shape')} dtype={d.get('dtype')} "
            f"quant={d.get('quantization')}"
        )

    print("\nOutputs:")
    for i, d in enumerate(outputs):
        print(
            f"  [{i}] name={d.get('name')} shape={d.get('shape')} dtype={d.get('dtype')} "
            f"quant={d.get('quantization')}"
        )

    # Show op list (doesn't always expose delegation info, but helpful for checking float fallbacks).
    try:
        ops = interpreter._get_ops_details()  # pylint: disable=protected-access
        op_names = [op.get("op_name", "<?>") for op in ops]
        uniq = sorted(set(op_names))
        print("\nOps (unique):")
        for name in uniq:
            print(f"  - {name} (count={op_names.count(name)})")
    except Exception as exc:
        print("\n[WARN] Could not read ops details:", exc)

    if args.analyze:
        print("\nAnalyzer output:")
        try:
            tf.lite.experimental.Analyzer.analyze(model_path=str(model_path))
        except Exception as exc:
            print("[WARN] Analyzer failed:", exc)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


