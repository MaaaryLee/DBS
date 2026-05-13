"""
Prepare a matching TFLite header + manifest for ESP32 benchmarking.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.convert_tflite_to_c import convert_tflite_to_c  # noqa: E402


def _default_model_path(variant: str) -> Path:
    if variant == "fp32":
        return ROOT / "tflite_actors/model_fp32.tflite"
    if variant == "int8":
        return ROOT / "tflite_actors/model_int8.tflite"
    return ROOT / variant


def _inspect_model(model_path: Path) -> Optional[Dict[str, object]]:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return None

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return {
        "input_shape": [int(value) for value in list(input_details["shape"])],
        "input_dtype": getattr(input_details["dtype"], "__name__", str(input_details["dtype"])),
        "input_quantization": [
            float(input_details.get("quantization", (0.0, 0))[0]),
            int(input_details.get("quantization", (0.0, 0))[1]),
        ],
        "output_shape": [int(value) for value in list(output_details["shape"])],
        "output_dtype": getattr(output_details["dtype"], "__name__", str(output_details["dtype"])),
        "output_quantization": [
            float(output_details.get("quantization", (0.0, 0))[0]),
            int(output_details.get("quantization", (0.0, 0))[1]),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ESP32 benchmark assets from a TFLite model.")
    parser.add_argument("--variant", choices=["fp32", "int8"], default="int8")
    parser.add_argument("--model-path", type=str, default=None, help="Override the .tflite model path")
    parser.add_argument("--root-header", type=str, default="model.h")
    parser.add_argument("--firmware-header", type=str, default="esp32_firmware/model.h")
    parser.add_argument("--manifest", type=str, default="esp32_firmware/model_manifest.json")
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else _default_model_path(args.variant)
    root_header = ROOT / args.root_header
    firmware_header = ROOT / args.firmware_header
    manifest_path = ROOT / args.manifest

    if not model_path.exists():
        raise SystemExit(f"missing model: {model_path}")

    converted = convert_tflite_to_c(
        tflite_path=str(model_path),
        output_path=str(root_header),
        var_name="dbs_model",
    )
    if not converted:
        raise SystemExit("failed to generate root model.h")

    firmware_header.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root_header, firmware_header)

    manifest = {
        "variant": args.variant,
        "model_path": str(model_path.relative_to(ROOT)),
        "root_header": str(root_header.relative_to(ROOT)),
        "firmware_header": str(firmware_header.relative_to(ROOT)),
        "model_size_bytes": model_path.stat().st_size,
        "header_size_bytes": firmware_header.stat().st_size,
        "model_info": _inspect_model(model_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 70)
    print("ESP32 BENCHMARK ASSETS READY")
    print("=" * 70)
    print(f"Variant          : {args.variant}")
    print(f"TFLite model     : {model_path}")
    print(f"Root header      : {root_header}")
    print(f"Firmware header  : {firmware_header}")
    print(f"Manifest         : {manifest_path}")
    if manifest["model_info"] is not None:
        info = manifest["model_info"]
        print(f"Input shape/dtype: {info['input_shape']} / {info['input_dtype']}")
        print(f"Output shape/dtype: {info['output_shape']} / {info['output_dtype']}")
    print("\nNext steps:")
    print("1. Compile the recommended serial benchmark sketch with:")
    print("   python3 scripts/compile_esp32_benchmark.py --upload")
    print("2. Run the serial benchmark helper:")
    print("   python3 scripts/run_esp32_benchmark.py --runs 200")
    print("3. If you want the older ArduTFLite on-device Hjorth pipeline benchmark instead:")
    print("   python3 scripts/compile_esp32_benchmark.py --mode legacy --upload")
    print("   python3 scripts/run_esp32_benchmark.py --mode legacy --timeout 30")
    print("4. Repeat with --variant fp32 and compare the BENCH_RESULT lines")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
