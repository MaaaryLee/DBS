import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _run_measure_script(
    *,
    script_name: str,
    model_path: str,
    states_path: str,
    runs: int,
    warmup: int,
    threads: int,
    inner_repeats: int,
    disable_default_delegates: bool,
    output_path: Path,
) -> Dict[str, object]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / script_name),
        "--model-path",
        model_path,
        "--states-path",
        states_path,
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        "--threads",
        str(threads),
        "--inner-repeats",
        str(inner_repeats),
        "--output-path",
        str(output_path),
    ]
    if disable_default_delegates:
        command.append("--disable-default-delegates")

    subprocess.run(command, check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    return json.loads(output_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare TFLite FP32 and INT8 using the standard measurement scripts.")
    parser.add_argument("--fp32-model", type=str, default="tflite_actors/model_fp32.tflite")
    parser.add_argument("--int8-model", type=str, default="tflite_actors/model_int8.tflite")
    parser.add_argument("--states-path", type=str, default="states_eval.npy")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inner-repeats", type=int, default=50)
    parser.add_argument("--output-path", type=str, default="results/tflite_comparison.json")
    args = parser.parse_args()

    temp_dir = ROOT / "results" / "tflite_compare_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for disable_default_delegates in (False, True):
        mode_name = "delegates_off" if disable_default_delegates else "delegates_on"
        rows.append(
            _run_measure_script(
                script_name="measure_tflite_fp32_latency.py",
                model_path=args.fp32_model,
                states_path=args.states_path,
                runs=args.runs,
                warmup=args.warmup,
                threads=args.threads,
                inner_repeats=args.inner_repeats,
                disable_default_delegates=disable_default_delegates,
                output_path=temp_dir / f"{mode_name}_fp32.json",
            )
        )
        rows.append(
            _run_measure_script(
                script_name="measure_tflite_int8_latency.py",
                model_path=args.int8_model,
                states_path=args.states_path,
                runs=args.runs,
                warmup=args.warmup,
                threads=args.threads,
                inner_repeats=args.inner_repeats,
                disable_default_delegates=disable_default_delegates,
                output_path=temp_dir / f"{mode_name}_int8.json",
            )
        )

    summary = {"rows": rows, "speedups": []}
    for delegates_enabled in (True, False):
        fp32_row = next(
            row for row in rows
            if row["dtype"] == "fp32" and row["default_delegates_enabled"] == delegates_enabled
        )
        int8_row = next(
            row for row in rows
            if row["dtype"] == "int8" and row["default_delegates_enabled"] == delegates_enabled
        )
        if int(fp32_row["obs_dim"]) != int(int8_row["obs_dim"]):
            raise SystemExit(
                "Refusing to compare FP32 and INT8 models with different input dimensions: "
                f"fp32 obs_dim={fp32_row['obs_dim']} vs int8 obs_dim={int8_row['obs_dim']}"
            )
        summary["speedups"].append(
            {
                "default_delegates_enabled": delegates_enabled,
                "obs_dim": int(fp32_row["obs_dim"]),
                "fp32_mean_ms": fp32_row["mean_ms"],
                "int8_mean_ms": int8_row["mean_ms"],
                "int8_speedup_vs_fp32": float(fp32_row["mean_ms"] / int8_row["mean_ms"]),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    print("=" * 70)
    print("TFLITE COMPARISON")
    print("=" * 70)
    for speedup in summary["speedups"]:
        mode = "delegates_on" if speedup["default_delegates_enabled"] else "delegates_off"
        print(
            f"{mode} (obs_dim={speedup['obs_dim']}): fp32={speedup['fp32_mean_ms']:.6f} ms | "
            f"int8={speedup['int8_mean_ms']:.6f} ms | "
            f"speedup={speedup['int8_speedup_vs_fp32']:.3f}x"
        )
    print("-" * 70)
    print(f"[OK] Wrote {output_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
