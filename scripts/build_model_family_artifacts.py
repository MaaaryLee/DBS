"""
Build the same export/benchmark artifact family we used for 96x96, but for any
TD3 actor hidden-layer size pair.

This script is intentionally orchestration-focused: it wires together the
existing training, quantization, TFLite export, ESP32 header generation, and
desktop latency benchmarks into one reproducible path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n[RUN] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def _model_label(h1: int, h2: int) -> str:
    return f"{h1}_{h2}"


def _results_dir(label: str) -> Path:
    return ROOT / "results" / "larger_models" / label


def _checkpoint_path(h1: int, h2: int, timesteps: int) -> Path:
    return ROOT / "models" / f"TD3_{h1}_{h2}" / f"{timesteps}.zip"


def _policy_artifact_paths(label: str) -> dict[str, Path]:
    out_dir = _results_dir(label)
    return {
        "onnx": out_dir / f"model_{label}.onnx",
        "tf_dir": out_dir / f"tf_model_{label}",
        "fp32_tflite": out_dir / f"model_fp32_{label}.tflite",
        "fp32_alias": out_dir / f"model_alias_fp32_{label}.tflite",
        "int8_tflite": out_dir / f"model_int8_{label}.tflite",
        "fp32_header": out_dir / f"model_fp32_{label}.h",
        "int8_header": out_dir / f"model_int8_{label}.h",
        "fp32_bench": out_dir / f"bench_fp32_{label}.json",
        "int8_bench": out_dir / f"bench_int8_{label}.json",
        "manifest": out_dir / f"pipeline_manifest_{label}.json",
    }


def _maybe_train_checkpoint(*, h1: int, h2: int, timesteps: int, seed: int, force: bool) -> Path:
    checkpoint = _checkpoint_path(h1, h2, timesteps)
    if checkpoint.exists() and not force:
        print(f"[OK] Reusing checkpoint: {checkpoint}")
        return checkpoint

    _run(
        [
            sys.executable,
            str(ROOT / "core" / "training.py"),
            "--h1",
            str(h1),
            "--h2",
            str(h2),
            "--timesteps",
            str(timesteps),
            "--checkpoints",
            "1",
            "--seed",
            str(seed),
        ]
    )
    if not checkpoint.exists():
        raise SystemExit(f"Expected training checkpoint was not created: {checkpoint}")
    return checkpoint


def _build_quantized_pytorch(*, h1: int, h2: int, timesteps: int, states_path: Path) -> None:
    _run(
        [
            sys.executable,
            str(ROOT / "core" / "quantize_model.py"),
            "--h1",
            str(h1),
            "--h2",
            str(h2),
            "--timesteps",
            str(timesteps),
            "--states-path",
            str(states_path),
        ]
    )


def _export_deployment_chain(
    *,
    checkpoint: Path,
    states_path: Path,
    output_paths: dict[str, Path],
) -> None:
    _run(
        [
            sys.executable,
            str(ROOT / "deployment" / "convert_to_onnx.py"),
            "--policy-path",
            str(checkpoint),
            "--output-path",
            str(output_paths["onnx"]),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "deployment" / "convert_onnx_to_tf.py"),
            "--onnx-path",
            str(output_paths["onnx"]),
            "--output-dir",
            str(output_paths["tf_dir"]),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "deployment" / "convert_tf_to_tflite.py"),
            "--saved-model-dir",
            str(output_paths["tf_dir"]),
            "--output-path",
            str(output_paths["fp32_tflite"]),
            "--alias-path",
            str(output_paths["fp32_alias"]),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "convert_saved_model_to_tflite_int8.py"),
            "--saved-model-dir",
            str(output_paths["tf_dir"]),
            "--output-path",
            str(output_paths["int8_tflite"]),
            "--states-path",
            str(states_path),
        ]
    )


def _export_esp32_headers(output_paths: dict[str, Path]) -> None:
    for key, tflite_key in (("fp32_header", "fp32_tflite"), ("int8_header", "int8_tflite")):
        _run(
            [
                sys.executable,
                str(ROOT / "deployment" / "convert_tflite_to_c.py"),
                "--tflite-path",
                str(output_paths[tflite_key]),
                "--output-path",
                str(output_paths[key]),
                "--var-name",
                "dbs_model",
            ]
        )


def _bench_tflite(output_paths: dict[str, Path], *, states_path: Path, runs: int, warmup: int, threads: int, inner_repeats: int) -> None:
    commands = [
        (
            ROOT / "scripts" / "measure_tflite_fp32_latency.py",
            output_paths["fp32_tflite"],
            output_paths["fp32_bench"],
        ),
        (
            ROOT / "scripts" / "measure_tflite_int8_latency.py",
            output_paths["int8_tflite"],
            output_paths["int8_bench"],
        ),
    ]
    for script_path, model_path, out_path in commands:
        _run(
            [
                sys.executable,
                str(script_path),
                "--model-path",
                str(model_path),
                "--states-path",
                str(states_path),
                "--runs",
                str(runs),
                "--warmup",
                str(warmup),
                "--threads",
                str(threads),
                "--inner-repeats",
                str(inner_repeats),
                "--output-path",
                str(out_path),
            ]
        )


def _inspect_tflite(path: Path) -> dict[str, object]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return {
        "size_bytes": path.stat().st_size,
        "input_shape": [int(v) for v in input_details["shape"]],
        "output_shape": [int(v) for v in output_details["shape"]],
        "input_dtype": getattr(input_details["dtype"], "__name__", str(input_details["dtype"])),
        "output_dtype": getattr(output_details["dtype"], "__name__", str(output_details["dtype"])),
        "input_quantization": [
            float(input_details.get("quantization", (0.0, 0))[0]),
            int(input_details.get("quantization", (0.0, 0))[1]),
        ],
        "output_quantization": [
            float(output_details.get("quantization", (0.0, 0))[0]),
            int(output_details.get("quantization", (0.0, 0))[1]),
        ],
    }


def _write_manifest(
    *,
    h1: int,
    h2: int,
    checkpoint: Path,
    states_path: Path,
    output_paths: dict[str, Path],
    built_steps: Iterable[str],
) -> None:
    manifest = {
        "label": _model_label(h1, h2),
        "hidden_layers": [h1, h2],
        "checkpoint": str(checkpoint.resolve()),
        "states_path": str(states_path.resolve()),
        "built_steps": list(built_steps),
        "artifacts": {key: str(path.resolve()) for key, path in output_paths.items()},
        "fp32_tflite_info": _inspect_tflite(output_paths["fp32_tflite"]) if output_paths["fp32_tflite"].exists() else None,
        "int8_tflite_info": _inspect_tflite(output_paths["int8_tflite"]) if output_paths["int8_tflite"].exists() else None,
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2))
    print(f"[OK] Wrote manifest: {output_paths['manifest']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the 96x96-style training/export/TFLite/ESP32 artifact flow for any TD3 actor size."
    )
    parser.add_argument("--h1", type=int, required=True, help="First hidden layer size")
    parser.add_argument("--h2", type=int, required=True, help="Second hidden layer size")
    parser.add_argument(
        "--checkpoint-timesteps",
        type=int,
        default=500,
        help="Training checkpoint timesteps to reuse or create, e.g. 500 or 1500",
    )
    parser.add_argument(
        "--states-path",
        type=str,
        default=None,
        help=(
            "Calibration / latency states base path. Dimension-specific files are handled automatically downstream. "
            "If omitted, the script keeps them local to results/larger_models/<h1>_<h2>/."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-if-missing", action="store_true", help="Train one checkpoint if it does not already exist.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if the checkpoint already exists.")
    parser.add_argument("--skip-quantize-pytorch", action="store_true")
    parser.add_argument("--skip-bench-tflite", action="store_true")
    parser.add_argument("--skip-esp32-headers", action="store_true")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inner-repeats", type=int, default=50)
    args = parser.parse_args()

    label = _model_label(args.h1, args.h2)
    out_dir = _results_dir(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _policy_artifact_paths(label)
    if args.states_path:
        states_path = Path(args.states_path).expanduser().resolve()
    else:
        # Keep per-model calibration / latency state files local to this artifact
        # family so we do not silently mutate repo-level benchmark assets.
        states_path = (out_dir / "states_eval.npy").resolve()

    checkpoint = _checkpoint_path(args.h1, args.h2, args.checkpoint_timesteps)
    if args.train_if_missing or args.force_retrain:
        checkpoint = _maybe_train_checkpoint(
            h1=args.h1,
            h2=args.h2,
            timesteps=args.checkpoint_timesteps,
            seed=args.seed,
            force=bool(args.force_retrain),
        )
    elif not checkpoint.exists():
        raise SystemExit(
            f"Missing checkpoint {checkpoint}. Re-run with --train-if-missing, "
            "or provide a matching models/TD3_<h1>_<h2>/<timesteps>.zip first."
        )

    built_steps: list[str] = []
    if not args.skip_quantize_pytorch:
        _build_quantized_pytorch(
            h1=args.h1,
            h2=args.h2,
            timesteps=args.checkpoint_timesteps,
            states_path=states_path,
        )
        built_steps.append("pytorch_quantization")

    _export_deployment_chain(
        checkpoint=checkpoint,
        states_path=states_path,
        output_paths=output_paths,
    )
    built_steps.extend(["onnx_export", "saved_model_export", "tflite_fp32_export", "tflite_int8_export"])

    if not args.skip_esp32_headers:
        _export_esp32_headers(output_paths)
        built_steps.append("esp32_headers")

    if not args.skip_bench_tflite:
        _bench_tflite(
            output_paths,
            states_path=states_path,
            runs=args.runs,
            warmup=args.warmup,
            threads=args.threads,
            inner_repeats=args.inner_repeats,
        )
        built_steps.append("desktop_tflite_bench")

    _write_manifest(
        h1=args.h1,
        h2=args.h2,
        checkpoint=checkpoint,
        states_path=states_path,
        output_paths=output_paths,
        built_steps=built_steps,
    )

    print("\n" + "=" * 70)
    print("MODEL FAMILY PIPELINE READY")
    print("=" * 70)
    print(f"Label            : {label}")
    print(f"Checkpoint       : {checkpoint}")
    print(f"Results dir      : {out_dir}")
    print(f"FP32 TFLite      : {output_paths['fp32_tflite']}")
    print(f"INT8 TFLite      : {output_paths['int8_tflite']}")
    print(f"FP32 ESP32 header: {output_paths['fp32_header']}")
    print(f"INT8 ESP32 header: {output_paths['int8_header']}")
    print("\nNext step if you want on-device native ESP32 benchmarking:")
    print(
        f"python3 {ROOT / 'scripts' / 'run_espidf_benchmark_variant.py'} "
        f"--model-path {output_paths['int8_tflite']} --label native_int8_{label} --port /dev/cu.usbmodem... --repeats 5"
    )
    print(
        f"python3 {ROOT / 'scripts' / 'run_espidf_benchmark_variant.py'} "
        f"--model-path {output_paths['fp32_tflite']} --label native_fp32_{label} --port /dev/cu.usbmodem... --repeats 5"
    )
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
