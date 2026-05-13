"""
Compile and run the standalone C++ cache-blocking matvec benchmark.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "cache_blocking_benchmark.cpp"
RESULTS_DIR = ROOT / "results" / "cache_blocking"


def _compile(binary_path: Path) -> None:
    cmd = [
        "clang++",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        str(SOURCE),
        "-o",
        str(binary_path),
    ]
    subprocess.run(cmd, check=True)


def _parse_output(stdout: str) -> dict:
    config: dict[str, str] = {}
    results: list[dict[str, object]] = []
    totals: list[dict[str, object]] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0]
        fields: dict[str, object] = {}
        for item in parts[1:]:
            key, value = item.split("=", 1)
            try:
                if "." in value:
                    fields[key] = float(value)
                else:
                    fields[key] = int(value)
            except ValueError:
                fields[key] = value
        if tag == "CONFIG":
            config = {key: str(value) for key, value in fields.items()}
        elif tag == "RESULT":
            results.append(fields)
        elif tag == "TOTAL":
            totals.append(fields)
        else:
            raise ValueError(f"Unexpected output line: {line}")

    if not config or not results or not totals:
        raise ValueError("Benchmark output was incomplete.")

    return {
        "config": config,
        "results": results,
        "totals": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone cache-blocking benchmark.")
    parser.add_argument("--preset", default="400x300", choices=("96x96", "400x300"))
    parser.add_argument("--row-tile", type=int, default=8)
    parser.add_argument("--col-tile", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--flush-bytes",
        type=int,
        default=0,
        help="Optional buffer touched before each run to reduce warm-cache bias.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    binary_path = Path("/tmp/dbs_cache_blocking_benchmark")
    _compile(binary_path)

    cmd = [
        str(binary_path),
        "--preset",
        args.preset,
        "--row-tile",
        str(args.row_tile),
        "--col-tile",
        str(args.col_tile),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--flush-bytes",
        str(args.flush_bytes),
    ]
    run = subprocess.run(cmd, check=True, capture_output=True, text=True)
    parsed = _parse_output(run.stdout)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else RESULTS_DIR / f"cache_blocking_{args.preset}_rt{args.row_tile}_ct{args.col_tile}_{timestamp}.json"
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(SOURCE),
        "stdout": run.stdout,
        **parsed,
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"[OK] Wrote cache-blocking benchmark to {output_path}")
    print()
    print("Per-layer results:")
    for item in parsed["results"]:
        print(
            f"  {item['layer']:>3} {item['kernel']:<18} "
            f"avg_ns={item['avg_ns']:.3f} ns_per_mac={item['ns_per_mac']:.6f} "
            f"gmac_per_s={item['gmac_per_s']:.6f} max_abs_diff={item['max_abs_diff']}"
        )
    print()
    print("Totals:")
    for item in parsed["totals"]:
        print(
            f"  {item['kernel']:<18} total_avg_ns={item['total_avg_ns']:.3f} "
            f"total_ns_per_mac={item['total_ns_per_mac']:.6f} "
            f"total_gmac_per_s={item['total_gmac_per_s']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
