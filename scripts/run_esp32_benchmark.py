"""
Send benchmark commands to the ESP32 firmware over serial and collect the result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict


def _parse_key_value_line(line: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for chunk in line.split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        fields[key] = value
    return fields


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True)


def _is_esp32_board_fqbn(fqbn: str) -> bool:
    return fqbn.startswith("esp32:esp32:") or fqbn.startswith("arduino:esp32:")


def _detect_serial_port() -> str | None:
    result = _run_command(["arduino-cli", "board", "list", "--format", "json"])
    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    candidates = []
    for item in payload.get("detected_ports", []):
        port = item.get("port", {})
        if port.get("protocol") != "serial":
            continue

        board_fqbns = [board.get("fqbn", "") for board in item.get("matching_boards", [])]
        if any(_is_esp32_board_fqbn(fqbn) for fqbn in board_fqbns):
            address = port.get("address")
            if address:
                candidates.append(address)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise SystemExit(f"Multiple ESP32 serial ports found: {', '.join(candidates)}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an ESP32 DBS firmware benchmark over serial.")
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/cu.usbserial-0001")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=["modern", "legacy"],
        default="modern",
        help="modern sends `bench`; legacy listens for the one-shot ArduTFLite benchmark output after boot",
    )
    parser.add_argument(
        "--observation",
        type=float,
        nargs="*",
        default=None,
        help="Optional observation values to send before benchmarking",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--save-log", default=None, help="Optional path to save raw serial output")
    parser.add_argument("--output-json", default=None, help="Optional path to save parsed benchmark JSON")
    args = parser.parse_args()

    try:
        import serial  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "pyserial is required for this helper. Install with: python3 -m pip install --user pyserial"
        ) from exc

    port = args.port or _detect_serial_port()
    if port is None:
        raise SystemExit("Could not auto-detect an ESP32 serial port. Re-run with --port /dev/cu.usbmodem...")

    log_lines = []
    with serial.Serial(port, args.baud, timeout=0.25) as ser:
        if args.mode == "modern":
            time.sleep(2.0)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        else:
            # The legacy ArduTFLite sketch prints its benchmark once after boot,
            # so avoid clearing the serial buffers here or we can discard it.
            time.sleep(0.25)

        if args.mode == "modern" and args.observation:
            values = " ".join(f"{value:.8g}" for value in args.observation)
            ser.write(f"sample {values}\n".encode("utf-8"))
            ser.flush()
            time.sleep(0.25)

        if args.mode == "modern":
            ser.write(f"bench {args.runs}\n".encode("utf-8"))
            ser.flush()

        started = time.time()
        last_result = None
        while time.time() - started < args.timeout:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            log_lines.append(raw)
            print(raw)
            if raw.startswith("BENCH_RESULT"):
                last_result = _parse_key_value_line(raw)
            if raw == "BENCH_DONE":
                break

    if last_result is None:
        raise SystemExit("did not receive a BENCH_RESULT line before timeout")

    runs_reported = last_result.get("runs")
    if runs_reported is not None:
        try:
            runs_reported_int = int(runs_reported)
        except ValueError:
            runs_reported_int = None
        if runs_reported_int is not None and runs_reported_int != int(args.runs):
            raise SystemExit(
                f"firmware reported runs={runs_reported_int}, but host requested --runs {args.runs}. "
                "The flashed benchmark configuration does not match the requested run count."
            )

    if args.save_log:
        log_path = Path(args.save_log).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n")

    parsed_result = {
        "port": port,
        "baud": args.baud,
        "mode": args.mode,
        "runs_requested": args.runs,
        "observation": args.observation,
        "bench_result": last_result,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(parsed_result, indent=2))

    print("\nParsed benchmark fields:")
    for key in sorted(last_result):
        print(f"{key}: {last_result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
