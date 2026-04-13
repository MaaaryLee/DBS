"""
Run the ESP32 benchmark helper multiple times and aggregate the results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_HELPER = ROOT / "scripts" / "run_esp32_benchmark.py"

NUMERIC_FIELDS = (
    "quant_avg_us",
    "invoke_avg_us",
    "dequant_avg_us",
    "total_avg_us",
    "min_invoke_us",
    "max_invoke_us",
)


def _to_number(value: str):
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def _aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "min": values[0], "max": values[0]}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeat ESP32 benchmark runs and aggregate the results.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mode", choices=["modern", "legacy"], default="legacy")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--label", required=True, help="Short label used for trace filenames")
    parser.add_argument("--out-dir", default="results/esp32/repeats")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    for index in range(args.repeats):
        stem = f"{args.label}_run{index + 1:02d}"
        json_path = out_dir / f"{stem}.json"
        log_path = out_dir / f"{stem}.log"
        command = [
            sys.executable,
            str(RUN_HELPER),
            "--mode",
            args.mode,
            "--runs",
            str(args.runs),
            "--baud",
            str(args.baud),
            "--timeout",
            str(args.timeout),
            "--save-log",
            str(log_path),
            "--output-json",
            str(json_path),
        ]
        if args.port:
            command.extend(["--port", args.port])

        print(f"[{index + 1}/{args.repeats}] {' '.join(command)}", flush=True)
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            raise SystemExit(f"Run {index + 1} failed with exit code {result.returncode}")

        payload = json.loads(json_path.read_text())
        bench = payload["bench_result"]
        run_summaries.append(
            {
                "run_index": index + 1,
                "json_path": str(json_path),
                "log_path": str(log_path),
                "bench_result": {key: _to_number(value) for key, value in bench.items()},
            }
        )

    aggregate = {}
    for field in NUMERIC_FIELDS:
        values = [
            float(run["bench_result"][field])
            for run in run_summaries
            if field in run["bench_result"]
        ]
        aggregate[field] = _aggregate(values)

    summary = {
        "label": args.label,
        "mode": args.mode,
        "runs_requested": args.runs,
        "repeats": args.repeats,
        "port": args.port,
        "aggregate": aggregate,
        "raw_runs": run_summaries,
    }
    summary_path = out_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nAggregate summary:")
    for field in NUMERIC_FIELDS:
        stats = aggregate[field]
        print(
            f"{field}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )
    print(f"\nSaved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
