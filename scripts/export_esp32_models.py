"""
Legacy helper to export both FP32 and INT8 TFLite models as C headers.

Prefer `scripts/prepare_esp32_benchmark.py` for the active benchmark workflow.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment"))

from convert_tflite_to_c import convert_tflite_to_c  # noqa: E402


EXPORTS = (
    ("tflite_actors/model_fp32.tflite", "esp32_firmware/model_fp32.h"),
    ("tflite_actors/model_int8.tflite", "esp32_firmware/model_int8.h"),
)


def main() -> int:
    print("Legacy export helper: prefer `python3 scripts/prepare_esp32_benchmark.py --variant int8|fp32`.")
    ok = True
    for source, target in EXPORTS:
        print(f"\nExporting {source} -> {target}")
        ok = convert_tflite_to_c(
            tflite_path=str(ROOT / source),
            output_path=str(ROOT / target),
            var_name="dbs_model",
        ) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
