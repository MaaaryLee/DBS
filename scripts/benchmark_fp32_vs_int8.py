"""
Unified benchmarking script: FP32 vs INT8 (PyTorch + optional TFLite).

Outputs:
- JSON + CSV table with latency + model size metrics
- Explicit speedups showing INT8 < FP32 latency (where applicable)

Notes on fairness:
- For PyTorch static INT8 actors, the default "full" path includes Quant/DeQuant stubs.
- The "core" path measures the quantized backbone with pre-quantized inputs, which better
  matches embedded deployment where inputs are quantized upstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for this benchmark.") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.quantize_model as quantize_model  # noqa: E402


# Ensure pickled modules saved under __main__ can be deserialized.
sys.modules["__main__"].TD3Actor = quantize_model.TD3Actor
sys.modules["__main__"].QuantizableTD3Actor = quantize_model.QuantizableTD3Actor


@dataclass
class BenchRow:
    backend: str  # pytorch / tflite
    dtype: str  # fp32 / int8
    mode: str  # full / core / tflite
    h1: int
    h2: int
    batch_size: int
    threads: int
    runs: int
    warmup: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    model_path: str
    model_size_kb: float
    notes: str = ""


def _percentile(arr: np.ndarray, pct: float) -> float:
    return float(np.percentile(arr, pct))


def _time_callable(fn, runs: int, warmup: int) -> np.ndarray:
    # perf_counter_ns has lower overhead and better resolution on Windows.
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return np.asarray(times, dtype=np.float64)


def _summarize(times_ms: np.ndarray) -> Dict[str, float]:
    return {
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std(ddof=0)),
        "p50_ms": _percentile(times_ms, 50),
        "p90_ms": _percentile(times_ms, 90),
        "min_ms": float(times_ms.min()),
        "max_ms": float(times_ms.max()),
    }


def _load_states(states_path: Path, batch_size: int) -> torch.Tensor:
    if not states_path.exists():
        raise SystemExit(f"missing {states_path}")
    states = np.load(states_path).astype("float32")
    if states.ndim != 2:
        raise SystemExit(f"expected states .npy to be 2D (N, obs_dim), got {states.shape}")
    batch = max(1, int(batch_size))
    if states.shape[0] < batch:
        raise SystemExit(f"states file has only {states.shape[0]} rows, need batch_size={batch}")
    return torch.from_numpy(states[:batch])


def _torch_threads(threads: int) -> None:
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(int(threads))
    except Exception:
        pass


def _pytorch_bench(
    *,
    h1: int,
    h2: int,
    states_path: Path,
    runs: int,
    warmup: int,
    batch_size: int,
    threads: int,
    variant: str,
) -> List[BenchRow]:
    _torch_threads(threads)

    fp32_path = Path(f"models/policies/actor_fp32_{h1}_{h2}.pt")
    if not fp32_path.exists():
        raise SystemExit(f"missing {fp32_path} (run: python core/quantize_model.py --h1 {h1} --h2 {h2} --timesteps <...>)")

    if variant == "static_int8":
        int8_path = Path(f"models/policies/actor_int8_static_{h1}_{h2}.pt")
    else:
        int8_path = Path(f"models/policies/actor_int8_dynamic_{h1}_{h2}.pt")

    if not int8_path.exists():
        raise SystemExit(f"missing {int8_path} (run: python core/quantize_model.py --h1 {h1} --h2 {h2} --timesteps <...>)")

    sample_fp32 = _load_states(states_path, batch_size=batch_size)

    fp32 = torch.load(fp32_path, map_location="cpu")
    fp32.eval()

    int8 = torch.load(int8_path, map_location="cpu")
    int8.eval()

    rows: List[BenchRow] = []

    with torch.inference_mode():
        # FP32
        times = _time_callable(lambda: fp32(sample_fp32), runs=runs, warmup=warmup)
        stats = _summarize(times)
        rows.append(
            BenchRow(
                backend="pytorch",
                dtype="fp32",
                mode="full",
                h1=h1,
                h2=h2,
                batch_size=batch_size,
                threads=threads,
                runs=runs,
                warmup=warmup,
                model_path=str(fp32_path),
                model_size_kb=float(fp32_path.stat().st_size / 1024.0),
                notes="",
                **stats,
            )
        )

        # INT8 (full)
        times = _time_callable(lambda: int8(sample_fp32), runs=runs, warmup=warmup)
        stats = _summarize(times)
        rows.append(
            BenchRow(
                backend="pytorch",
                dtype="int8",
                mode="full",
                h1=h1,
                h2=h2,
                batch_size=batch_size,
                threads=threads,
                runs=runs,
                warmup=warmup,
                model_path=str(int8_path),
                model_size_kb=float(int8_path.stat().st_size / 1024.0),
                notes=f"variant={variant}",
                **stats,
            )
        )

        # INT8 (core-only): only meaningful for static quant with quantized layers.
        if variant == "static_int8":
            backbone = getattr(int8, "backbone", None)
            first_layer = None
            if backbone is not None and hasattr(backbone, "__getitem__"):
                try:
                    first_layer = backbone[0]
                except Exception:
                    first_layer = None

            if first_layer is not None and hasattr(first_layer, "scale") and hasattr(first_layer, "zero_point"):
                qsample = torch.quantize_per_tensor(
                    sample_fp32,
                    scale=float(first_layer.scale),
                    zero_point=int(first_layer.zero_point),
                    dtype=torch.quint8,
                )
                times = _time_callable(lambda: backbone(qsample), runs=runs, warmup=warmup)
                stats = _summarize(times)
                rows.append(
                    BenchRow(
                        backend="pytorch",
                        dtype="int8",
                        mode="core",
                        h1=h1,
                        h2=h2,
                        batch_size=batch_size,
                        threads=threads,
                        runs=runs,
                        warmup=warmup,
                        model_path=str(int8_path),
                        model_size_kb=float(int8_path.stat().st_size / 1024.0),
                        notes="static_int8 backbone only; inputs pre-quantized",
                        **stats,
                    )
                )
            else:
                # Still return full mode; core mode isn't available.
                pass

    return rows


def _tflite_bench(
    *,
    model_path: Path,
    states_path: Path,
    runs: int,
    warmup: int,
    batch_size: int,
    threads: int,
    dtype: str,
) -> BenchRow:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("TensorFlow is required for TFLite benchmarking.") from exc

    import os

    if not model_path.exists():
        raise SystemExit(f"missing {model_path}")
    if not states_path.exists():
        raise SystemExit(f"missing {states_path}")

    disable_default_delegates = os.environ.get("TFLITE_DISABLE_DEFAULT_DELEGATES", "1") == "1"
    interpreter_kwargs = {"model_path": str(model_path), "num_threads": int(threads)}
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
    locked_shape = [int(batch_size)] + list(input_details[0]["shape"][1:])

    # Only resize if needed (avoids issues with delegates requiring static tensors).
    if list(input_details[0]["shape"]) != locked_shape:
        interpreter.resize_tensor_input(input_index, locked_shape)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()

    states = np.load(states_path).astype("float32")
    if states.shape[0] < int(batch_size):
        raise SystemExit(f"states file has only {states.shape[0]} rows, need batch_size={batch_size}")
    sample = states[: int(batch_size)].reshape(locked_shape)

    notes = ""
    if dtype == "int8":
        input_scale, input_zero_point = input_details[0]["quantization"]
        if input_scale == 0:
            raise SystemExit("input scale is zero; quantization info missing in TFLite model")
        sample = np.round(sample / input_scale + input_zero_point).astype(np.int8)
        notes = "input pre-quantized using model input quantization params"

    def invoke():
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()

    times = _time_callable(invoke, runs=runs, warmup=warmup)
    stats = _summarize(times)

    return BenchRow(
        backend="tflite",
        dtype=dtype,
        mode="tflite",
        h1=-1,
        h2=-1,
        batch_size=batch_size,
        threads=threads,
        runs=runs,
        warmup=warmup,
        model_path=str(model_path),
        model_size_kb=float(model_path.stat().st_size / 1024.0),
        notes=notes,
        **stats,
    )


def _write_csv(path: Path, rows: List[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _speedups(rows: List[BenchRow]) -> List[Dict[str, Any]]:
    # Compute speedup per backend relative to fp32 (same backend, same mode=full/tflite).
    by_key: Dict[Tuple[str, str], BenchRow] = {}
    for r in rows:
        # baseline key: backend + (full or tflite)
        base_mode = "tflite" if r.backend == "tflite" else "full"
        key = (r.backend, base_mode)
        if r.dtype == "fp32" and r.mode == base_mode:
            by_key[key] = r

    results: List[Dict[str, Any]] = []
    for r in rows:
        base_mode = "tflite" if r.backend == "tflite" else "full"
        base = by_key.get((r.backend, base_mode))
        if not base:
            continue
        results.append(
            {
                "backend": r.backend,
                "dtype": r.dtype,
                "mode": r.mode,
                "mean_ms": r.mean_ms,
                "baseline_fp32_mean_ms": base.mean_ms,
                "speedup_vs_fp32": float(base.mean_ms / r.mean_ms) if r.mean_ms > 0 else None,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs INT8 inference time + model size.")
    parser.add_argument("--h1", type=int, default=22)
    parser.add_argument("--h2", type=int, default=22)
    parser.add_argument("--states-path", type=str, default="states_eval.npy")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--variant", choices=["static_int8", "dynamic_int8"], default="static_int8")
    parser.add_argument("--include-tflite", action="store_true", help="Also benchmark TFLite FP32/INT8 models")
    parser.add_argument("--tflite-fp32-path", type=str, default="tflite_actors/model_fp32.tflite")
    parser.add_argument("--tflite-int8-path", type=str, default="tflite_actors/model_int8.tflite")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_latest")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states_path = Path(args.states_path)
    rows: List[BenchRow] = []

    # PyTorch
    rows.extend(
        _pytorch_bench(
            h1=args.h1,
            h2=args.h2,
            states_path=states_path,
            runs=int(args.runs),
            warmup=int(args.warmup),
            batch_size=int(args.batch_size),
            threads=int(args.threads),
            variant=args.variant,
        )
    )

    # Optional: TFLite
    if args.include_tflite:
        rows.append(
            _tflite_bench(
                model_path=Path(args.tflite_fp32_path),
                states_path=states_path,
                runs=int(args.runs),
                warmup=int(args.warmup),
                batch_size=int(args.batch_size),
                threads=int(args.threads),
                dtype="fp32",
            )
        )
        rows.append(
            _tflite_bench(
                model_path=Path(args.tflite_int8_path),
                states_path=states_path,
                runs=int(args.runs),
                warmup=int(args.warmup),
                batch_size=int(args.batch_size),
                threads=int(args.threads),
                dtype="int8",
            )
        )

    # Write outputs
    json_path = output_dir / "benchmark_rows.json"
    csv_path = output_dir / "benchmark_rows.csv"
    speedup_path = output_dir / "benchmark_speedups.json"

    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    _write_csv(csv_path, rows)
    speedup_path.write_text(json.dumps(_speedups(rows), indent=2))

    # Console summary (high-signal)
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for item in _speedups(rows):
        if item["dtype"] == "fp32" and item["mode"] in ("full", "tflite"):
            continue
        print(
            f"{item['backend']:>7} | {item['dtype']:<4} | {item['mode']:<5} | "
            f"mean={item['mean_ms']:.4f} ms | "
            f"speedup_vs_fp32={item['speedup_vs_fp32']:.2f}x"
        )
    print("-" * 70)
    print(f"[OK] Wrote {json_path}")
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {speedup_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


