"""
Compile the isolated ESP32 benchmark sketch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKETCH = ROOT / "esp32_firmware/dbs_benchmark"
LEGACY_SKETCH = ROOT / "esp32_firmware/dbs_ardutflite_benchmark"
DEFAULT_FQBN = "esp32:esp32:nano_nora"
KNOWN_SPAN_ERROR = "assignment of read-only member 'flatbuffers::span<T, Extent>::count_'"


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def _is_esp32_board_fqbn(fqbn: str) -> bool:
    return fqbn.startswith("esp32:esp32:") or fqbn.startswith("arduino:esp32:")


def detect_serial_port(preferred_fqbn: str) -> str | None:
    result = run_command(["arduino-cli", "board", "list", "--format", "json"], ROOT)
    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    matched_ports: list[str] = []
    fallback_ports: list[str] = []
    for item in payload.get("detected_ports", []):
        port = item.get("port", {})
        if port.get("protocol") != "serial":
            continue

        address = port.get("address")
        if not address:
            continue

        board_fqbns = [board.get("fqbn", "") for board in item.get("matching_boards", [])]
        preferred_board_id = preferred_fqbn.split(":")[-1]
        if preferred_fqbn in board_fqbns or any(
            fqbn.split(":")[-1] == preferred_board_id for fqbn in board_fqbns
        ):
            matched_ports.append(address)
        elif any(_is_esp32_board_fqbn(fqbn) for fqbn in board_fqbns):
            fallback_ports.append(address)

    if len(matched_ports) == 1:
        return matched_ports[0]
    if len(matched_ports) > 1:
        raise SystemExit(f"Multiple matching ESP32 serial ports found: {', '.join(matched_ports)}")
    if len(fallback_ports) == 1:
        return fallback_ports[0]
    if len(fallback_ports) > 1:
        raise SystemExit(f"Multiple ESP32 serial ports found: {', '.join(fallback_ports)}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile the isolated ESP32 DBS benchmark sketch.")
    parser.add_argument("--fqbn", default=DEFAULT_FQBN, help="Arduino board FQBN")
    parser.add_argument("--sketch", default=str(DEFAULT_SKETCH), help="Sketch directory to compile")
    parser.add_argument(
        "--mode",
        choices=["modern", "legacy"],
        default="modern",
        help="modern=dbs_inference serial benchmark, legacy=ArduTFLite Hjorth pipeline benchmark",
    )
    parser.add_argument("--upload", action="store_true", help="Upload to the board after a successful compile")
    parser.add_argument("--port", default=None, help="Serial port to upload to")
    parser.add_argument(
        "--patch-tensorflowlite-esp32",
        action="store_true",
        help="Apply the legacy TensorFlowLite_ESP32 compatibility patch before compiling",
    )
    args = parser.parse_args()

    default_sketch = LEGACY_SKETCH if args.mode == "legacy" else DEFAULT_SKETCH
    sketch_dir = Path(args.sketch).expanduser().resolve()
    if args.sketch == str(DEFAULT_SKETCH):
        sketch_dir = default_sketch
    if not sketch_dir.exists():
        raise SystemExit(f"Sketch directory not found: {sketch_dir}")

    if args.patch_tensorflowlite_esp32:
        patch_cmd = [sys.executable, str(ROOT / "scripts/patch_tensorflowlite_esp32.py")]
        patch_result = run_command(patch_cmd, ROOT)
        if patch_result.stdout:
            print(patch_result.stdout, end="")
        if patch_result.stderr:
            print(patch_result.stderr, file=sys.stderr, end="")
        if patch_result.returncode != 0:
            return patch_result.returncode

    compile_cmd = ["arduino-cli", "compile", "--fqbn", args.fqbn, str(sketch_dir)]
    result = run_command(compile_cmd, ROOT)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    if result.returncode != 0 and KNOWN_SPAN_ERROR in result.stderr and not args.patch_tensorflowlite_esp32:
        print(
            "\nDetected the known TensorFlowLite_ESP32 FlatBuffers compatibility issue.\n"
            "If you switch back to that library, re-run with:\n"
            "python3 scripts/compile_esp32_benchmark.py --patch-tensorflowlite-esp32",
            file=sys.stderr,
        )
        return result.returncode

    if result.returncode != 0:
        return result.returncode

    if not args.upload:
        return 0

    port = args.port or detect_serial_port(args.fqbn)
    if port is None:
        raise SystemExit(
            "Could not auto-detect an ESP32 serial port. Re-run with --port /dev/cu.usbmodem..."
        )

    upload_cmd = ["arduino-cli", "upload", "-p", port, "--fqbn", args.fqbn, str(sketch_dir)]
    upload_result = run_command(upload_cmd, ROOT)
    if upload_result.stdout:
        print(upload_result.stdout, end="")
    if upload_result.stderr:
        print(upload_result.stderr, file=sys.stderr, end="")
    if upload_result.returncode == 0:
        print(f"Upload complete: {port}")
    return upload_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
