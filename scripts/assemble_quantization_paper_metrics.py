"""
Assemble a paper-ready quantization evidence bundle with trace paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "paper_quantization"

SOURCE_URLS = {
    "arduino_nano_esp32": "https://docs.arduino.cc/hardware/nano-esp32",
    "esp_tflite_micro": "https://github.com/espressif/esp-tflite-micro",
    "esp_nn": "https://github.com/espressif/esp-nn",
    "tensorflow_integer_quantization": "https://www.tensorflow.org/lite/performance/post_training_integer_quant",
}


def _abs(path: str) -> str:
    return str((ROOT / path).resolve())


def _load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def _mlp_stats(h1: int, h2: int, obs_dim: int = 6, act_dim: int = 2) -> dict[str, int]:
    params = obs_dim * h1 + h1 + h1 * h2 + h2 + h2 * act_dim + act_dim
    macs = obs_dim * h1 + h1 * h2 + h2 * act_dim
    return {
        "params": params,
        "macs": macs,
        "fp32_parameter_bytes": params * 4,
        "int8_parameter_bytes": params,
    }


def _latest_build_log() -> str:
    log_dir = ROOT / "espidf_firmware" / "dbs_espnn_benchmark" / "build" / "log"
    logs = sorted(log_dir.glob("idf_py_stdout_output_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return ""
    return str(logs[0].resolve())


def _state_summary(path: str) -> dict[str, Any]:
    arr = np.load(ROOT / path).astype(np.float32)
    per_dim_std = np.std(arr, axis=0)
    return {
        "path": _abs(path),
        "shape": [int(v) for v in arr.shape],
        "unique_rows": int(len(np.unique(arr, axis=0))),
        "per_dim_std": [float(v) for v in per_dim_std],
        "per_dim_min": [float(v) for v in np.min(arr, axis=0)],
        "per_dim_max": [float(v) for v in np.max(arr, axis=0)],
    }


def _screening_entry(label: str) -> dict[str, Any]:
    h1, h2 = map(int, label.split("_"))
    fp32_summary = _load_json(f"results/esp32/repeats/native_fp32_{label}_summary.json")
    repaired_int8_summary_path = ROOT / f"results/esp32/repeats/native_int8_{label}_repaired_32k_summary.json"
    int8_summary_rel = (
        f"results/esp32/repeats/native_int8_{label}_repaired_32k_summary.json"
        if repaired_int8_summary_path.exists()
        else f"results/esp32/repeats/native_int8_{label}_summary.json"
    )
    int8_summary = _load_json(int8_summary_rel)
    fp32_bench = _load_json(f"results/larger_models/{label}/bench_fp32_{label}.json")
    repaired_int8_bench_path = ROOT / f"results/larger_models/{label}/bench_int8_{label}_repaired.json"
    int8_bench_rel = (
        f"results/larger_models/{label}/bench_int8_{label}_repaired.json"
        if repaired_int8_bench_path.exists()
        else f"results/larger_models/{label}/bench_int8_{label}.json"
    )
    int8_bench = _load_json(int8_bench_rel)
    stats = _mlp_stats(h1, h2)
    fp32_invoke = fp32_summary["aggregate"]["invoke_avg_us"]["mean"]
    int8_invoke = int8_summary["aggregate"]["invoke_avg_us"]["mean"]
    return {
        "model_size": label,
        "hidden_layers": [h1, h2],
        **stats,
        "desktop_tflite_mean_ms": {
            "fp32": fp32_bench["mean_ms"],
            "int8": int8_bench["mean_ms"],
            "int8_speedup_vs_fp32": fp32_bench["mean_ms"] / int8_bench["mean_ms"],
        },
        "esp32_default_cache_32kb_invoke_us": {
            "fp32_mean": fp32_invoke,
            "fp32_std": fp32_summary["aggregate"]["invoke_avg_us"]["std"],
            "int8_mean": int8_invoke,
            "int8_std": int8_summary["aggregate"]["invoke_avg_us"]["std"],
            "int8_speedup_vs_fp32": fp32_invoke / int8_invoke,
        },
        "trace_paths": {
            "desktop_fp32": _abs(f"results/larger_models/{label}/bench_fp32_{label}.json"),
            "desktop_int8": _abs(int8_bench_rel),
            "esp32_fp32": _abs(f"results/esp32/repeats/native_fp32_{label}_summary.json"),
            "esp32_int8": _abs(int8_summary_rel),
        },
    }


def _candidate_section() -> dict[str, Any]:
    fp32_esp32 = _load_json("results/esp32/repeats/native_fp32_96_96_summary.json")
    int8_esp32_repaired = _load_json("results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json")
    fp32_esp32_dc64 = _load_json("results/esp32/repeats/native_fp32_96_96_dc64k_summary.json")
    int8_esp32_dc64 = _load_json("results/esp32/repeats/native_int8_96_96_dc64k_summary.json")
    fp32_desktop = _load_json("results/larger_models/96_96/bench_fp32_96_96_repaired.json")
    int8_desktop = _load_json("results/larger_models/96_96/bench_int8_96_96_repaired.json")
    eval_repaired = _load_json("results/larger_models/96_96/eval_tflite_96_96_repaired.json")
    stats = _mlp_stats(96, 96)

    fp32_32k_invoke = fp32_esp32["aggregate"]["invoke_avg_us"]["mean"]
    int8_32k_invoke = int8_esp32_repaired["aggregate"]["invoke_avg_us"]["mean"]
    fp32_64k_invoke = fp32_esp32_dc64["aggregate"]["invoke_avg_us"]["mean"]
    int8_64k_invoke = int8_esp32_dc64["aggregate"]["invoke_avg_us"]["mean"]

    return {
        "model_size": "96_96",
        "hidden_layers": [96, 96],
        **stats,
        "desktop_tflite_repaired": {
            "fp32_mean_ms": fp32_desktop["mean_ms"],
            "int8_mean_ms": int8_desktop["mean_ms"],
            "int8_speedup_vs_fp32": fp32_desktop["mean_ms"] / int8_desktop["mean_ms"],
            "fp32_trace": _abs("results/larger_models/96_96/bench_fp32_96_96_repaired.json"),
            "int8_trace": _abs("results/larger_models/96_96/bench_int8_96_96_repaired.json"),
        },
        "esp32_native_default_cache_32kb": {
            "fp32_invoke_mean_us": fp32_32k_invoke,
            "fp32_invoke_std_us": fp32_esp32["aggregate"]["invoke_avg_us"]["std"],
            "int8_invoke_mean_us": int8_32k_invoke,
            "int8_invoke_std_us": int8_esp32_repaired["aggregate"]["invoke_avg_us"]["std"],
            "int8_speedup_vs_fp32": fp32_32k_invoke / int8_32k_invoke,
            "fp32_trace": _abs("results/esp32/repeats/native_fp32_96_96_summary.json"),
            "int8_trace": _abs("results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json"),
        },
        "esp32_cache_ablation": {
            "data_cache_32kb": {
                "fp32_invoke_mean_us": fp32_32k_invoke,
                "int8_invoke_mean_us": int8_32k_invoke,
            },
            "data_cache_64kb": {
                "fp32_invoke_mean_us": fp32_64k_invoke,
                "int8_invoke_mean_us": int8_64k_invoke,
            },
            "interpretation": (
                "At 96x96, INT8 is faster than FP32 with the project's default 32 KB data cache, "
                "but FP32 becomes faster again when the data cache is increased to 64 KB."
            ),
            "trace_paths": {
                "fp32_32kb": _abs("results/esp32/repeats/native_fp32_96_96_summary.json"),
                "int8_32kb_repaired": _abs("results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json"),
                "fp32_64kb": _abs("results/esp32/repeats/native_fp32_96_96_dc64k_summary.json"),
                "int8_64kb": _abs("results/esp32/repeats/native_int8_96_96_dc64k_summary.json"),
            },
        },
        "quantization_fidelity_repaired": {
            "trace": _abs("results/larger_models/96_96/eval_tflite_96_96_repaired.json"),
            "model_size_bytes": eval_repaired["model_size_bytes"],
            "fp32_io": eval_repaired["fp32_io"],
            "int8_io": eval_repaired["int8_io"],
            "input_saturation": eval_repaired["input_saturation"],
            "state_coverage": eval_repaired["state_coverage"],
            "output_coverage": eval_repaired["output_coverage"],
            "fidelity": eval_repaired["fidelity"],
            "env_eval": eval_repaired["env_eval"],
            "warnings": eval_repaired["warnings"],
        },
        "calibration_state_sets": {
            "original_degenerate": _state_summary("states_eval_6d.npy"),
            "repaired_windowed": _state_summary("states_eval_6d_repaired.npy"),
        },
    }


def _not_right_list() -> list[dict[str, Any]]:
    return [
        {
            "status": "fixed for the candidate 96x96 rerun",
            "severity": "high",
            "issue": "The original 6D representative dataset collapsed to one repeated state.",
            "evidence": [
                _abs("states_eval_6d.npy"),
                _abs("results/larger_models/96_96/eval_tflite_96_96.json"),
                _abs("results/larger_models/96_96/eval_tflite_96_96_repaired.json"),
            ],
            "impact": "It invalidated the original 6D fidelity/correlation analysis and likely distorted INT8 calibration ranges.",
        },
        {
            "status": "open limitation",
            "severity": "high",
            "issue": "The cached 6D environment is deterministic in offline mode.",
            "evidence": [
                _abs("core/BGN_MC_Online.py"),
                _abs("results/larger_models/96_96/eval_tflite_96_96_repaired.json"),
            ],
            "impact": "The env_eval block is only a functional smoke test; it is not strong evidence for stochastic control performance.",
        },
        {
            "status": "open limitation",
            "severity": "medium",
            "issue": "The repaired 6D fidelity evaluation uses the same state-set family that was also used for INT8 calibration.",
            "evidence": [
                _abs("states_eval_6d_repaired.npy"),
                _abs("results/larger_models/96_96/eval_tflite_96_96_repaired.json"),
                _abs("scripts/convert_saved_model_to_tflite_int8.py"),
            ],
            "impact": "The fidelity numbers are useful and quantifiable, but they should be described as in-distribution or calibration-family fidelity unless a held-out 6D state set is added.",
        },
        {
            "status": "open nuance",
            "severity": "high",
            "issue": "The 96x96 INT8 speedup on ESP32-S3 depends on the default 32 KB data-cache configuration.",
            "evidence": [
                _abs("results/esp32/repeats/native_fp32_96_96_summary.json"),
                _abs("results/esp32/repeats/native_fp32_96_96_dc64k_summary.json"),
                _abs("results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json"),
                _abs("results/esp32/repeats/native_int8_96_96_dc64k_summary.json"),
                _abs("espidf_firmware/dbs_espnn_benchmark/sdkconfig"),
            ],
            "impact": "The paper must not claim a hardware-wide INT8 win without stating the cache/runtime configuration.",
        },
        {
            "status": "updated after repaired rerun",
            "severity": "medium",
            "issue": "The 80x80 and 128x128 threshold-screening models were rerun with repaired INT8 calibration, but they still only have latency-screening traces rather than full repaired fidelity/control evaluations.",
            "evidence": [
                _abs("results/esp32/repeats/native_int8_80_80_repaired_32k_summary.json"),
                _abs("results/esp32/repeats/native_int8_128_128_repaired_32k_summary.json"),
                _abs("results/larger_models/80_80/bench_int8_80_80_repaired.json"),
                _abs("results/larger_models/128_128/bench_int8_128_128_repaired.json"),
                _abs("states_eval_6d_repaired.npy"),
            ],
            "impact": "The repaired threshold crossover is valid as a latency result, but only the 96x96 candidate currently has the full repaired fidelity/control evidence bundle.",
        },
        {
            "status": "fixed",
            "severity": "medium",
            "issue": "The desktop TFLite latency helpers could silently append another `_6d` suffix and benchmark a synthesized fallback file.",
            "evidence": [
                _abs("scripts/measure_tflite_fp32_latency.py"),
                _abs("scripts/measure_tflite_int8_latency.py"),
                _abs("results/larger_models/96_96/bench_fp32_96_96_repaired.json"),
                _abs("results/larger_models/96_96/bench_int8_96_96_repaired.json"),
            ],
            "impact": "The repaired desktop traces are valid; older traces should be checked if they relied on custom state-file names.",
        },
    ]


def _report() -> dict[str, Any]:
    screening_labels = ["80_80", "96_96", "128_128"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT.resolve()),
        "hardware_runtime": {
            "board": "Arduino Nano ESP32",
            "chip": "ESP32-S3",
            "chip_trace_path": _latest_build_log(),
            "espidf_project": _abs("espidf_firmware/dbs_espnn_benchmark"),
            "current_sdkconfig_path": _abs("espidf_firmware/dbs_espnn_benchmark/sdkconfig"),
            "source_urls": SOURCE_URLS,
        },
        "candidate_model": _candidate_section(),
        "screening_latency_study": {
            "scope": "6D MLP latency screening under the default 32 KB ESP32-S3 data-cache configuration",
            "models": [_screening_entry(label) for label in screening_labels],
            "first_tested_int8_winner_under_default_cache": "96_96",
        },
        "not_right": _not_right_list(),
    }


def _markdown(report: dict[str, Any]) -> str:
    candidate = report["candidate_model"]
    desktop = candidate["desktop_tflite_repaired"]
    esp32 = candidate["esp32_native_default_cache_32kb"]
    cache = candidate["esp32_cache_ablation"]
    fidelity = candidate["quantization_fidelity_repaired"]

    lines = [
        "# Quantization Evidence Bundle",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Candidate 96x96 (6D)",
        "",
        f"- Params: {candidate['params']}",
        f"- MACs per inference: {candidate['macs']}",
        f"- FP32 parameter bytes: {candidate['fp32_parameter_bytes']}",
        f"- INT8 parameter bytes: {candidate['int8_parameter_bytes']}",
        f"- Model size reduction: {fidelity['model_size_bytes']['reduction_percent']:.2f}%",
        f"- Desktop TFLite latency: FP32 {desktop['fp32_mean_ms']:.9f} ms, INT8 {desktop['int8_mean_ms']:.9f} ms, INT8 speedup {desktop['int8_speedup_vs_fp32']:.3f}x",
        f"- ESP32 default-cache latency: FP32 {esp32['fp32_invoke_mean_us']:.3f} us, INT8 {esp32['int8_invoke_mean_us']:.3f} us, INT8 speedup {esp32['int8_speedup_vs_fp32']:.3f}x",
        f"- ESP32 64 KB cache latency: FP32 {cache['data_cache_64kb']['fp32_invoke_mean_us']:.3f} us, INT8 {cache['data_cache_64kb']['int8_invoke_mean_us']:.3f} us",
        f"- Repaired state coverage: {fidelity['state_coverage']['unique_rows']} unique states",
        f"- Fidelity MAE: {fidelity['fidelity']['mae']:.9f}",
        f"- Fidelity max abs diff: {fidelity['fidelity']['max_abs_diff']:.9f}",
        "",
        "## Threshold Screening",
        "",
        "| Model | Params | FP32 bytes | INT8 bytes | ESP32 FP32 us | ESP32 INT8 us | INT8 speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for entry in report["screening_latency_study"]["models"]:
        esp_entry = entry["esp32_default_cache_32kb_invoke_us"]
        lines.append(
            f"| {entry['model_size']} | {entry['params']} | {entry['fp32_parameter_bytes']} | "
            f"{entry['int8_parameter_bytes']} | {esp_entry['fp32_mean']:.3f} | "
            f"{esp_entry['int8_mean']:.3f} | {esp_entry['int8_speedup_vs_fp32']:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Trace Files",
            "",
            f"- Candidate desktop FP32: `{desktop['fp32_trace']}`",
            f"- Candidate desktop INT8: `{desktop['int8_trace']}`",
            f"- Candidate ESP32 FP32: `{esp32['fp32_trace']}`",
            f"- Candidate ESP32 INT8: `{esp32['int8_trace']}`",
            f"- Candidate fidelity: `{fidelity['trace']}`",
            f"- Original 6D states: `{candidate['calibration_state_sets']['original_degenerate']['path']}`",
            f"- Repaired 6D states: `{candidate['calibration_state_sets']['repaired_windowed']['path']}`",
            "",
            "## Things That Need Caution",
            "",
        ]
    )

    for item in report["not_right"]:
        lines.append(f"- [{item['severity']}/{item['status']}] {item['issue']} Impact: {item['impact']}")

    lines.extend(
        [
            "",
            "## Source Links",
            "",
            f"- Arduino Nano ESP32: {SOURCE_URLS['arduino_nano_esp32']}",
            f"- esp-tflite-micro: {SOURCE_URLS['esp_tflite_micro']}",
            f"- ESP-NN: {SOURCE_URLS['esp_nn']}",
            f"- TensorFlow Lite integer quantization: {SOURCE_URLS['tensorflow_integer_quantization']}",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    report = _report()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "quantization_paper_metrics.json"
    md_path = OUT_DIR / "quantization_paper_metrics.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
