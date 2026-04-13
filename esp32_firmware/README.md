# ESP32 DBS Inference Firmware

This directory contains the on-device ESP32 benchmark path for the DBS actor.

Current setup:

- `dbs_inference.ino` loads the active `model.h`, detects tensor shapes and dtypes at runtime, and benchmarks `quantize`, `invoke`, `dequant`, and `total`.
- The Arduino entry sketch is `dbs_benchmark/dbs_benchmark.ino`, which keeps the build isolated from unrelated `.ino` files in this folder.
- `sketch_dec4b.ino` is the older ArduTFLite Hjorth-feature pipeline benchmark, and its Arduino entry sketch is `dbs_ardutflite_benchmark/dbs_ardutflite_benchmark.ino`.
- The firmware now uses `Chirale_TensorFLowLite`, which compiles cleanly with the installed `esp32:esp32` core on this machine.
- The resolver is trimmed to the two ops this actor uses: `FULLY_CONNECTED` and `TANH`.

## Files

- `dbs_inference.ino` - Main firmware with serial commands and timing output
- `dbs_benchmark/dbs_benchmark.ino` - Arduino sketch entry point for compile/upload
- `sketch_dec4b.ino` - Older ArduTFLite Hjorth benchmark that computes 4 features on-device
- `dbs_ardutflite_benchmark/dbs_ardutflite_benchmark.ino` - Arduino sketch entry point for the older ArduTFLite benchmark
- `model.h` - Active generated model header consumed by the firmware
- `model_manifest.json` - Summary of the currently prepared model variant

## Which Arduino Sketch To Open

If you are using Arduino IDE:

- Open `/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_benchmark/dbs_benchmark.ino` if you want the recommended benchmark path with explicit `quantize` / `invoke` / `dequant` timing and flexible input dimension support.
- Open `/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_ardutflite_benchmark/dbs_ardutflite_benchmark.ino` only if you specifically want the older ArduTFLite Hjorth pipeline benchmark that computes the 4D features on-device before inference.

For your current situation, the safer old-sketch replacement is:

- `/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_ardutflite_benchmark/dbs_ardutflite_benchmark.ino`

That path now:

- uses the current generated model header through `model_int_8.h`
- checks whether the flashed model really expects 4 inputs
- reports a clear error if the model dimension and the sketch do not match
- prints a machine-readable `BENCH_RESULT` line

## Recommended Workflow

### 1. Install dependencies

In Arduino IDE or Arduino CLI, install:

- `esp32:esp32`
- `Chirale_TensorFLowLite`

CLI example:

```bash
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "Chirale_TensorFLowLite"
```

### 2. Prepare the model variant

From the project root:

```bash
python3 scripts/prepare_esp32_benchmark.py --variant int8
```

This refreshes:

- `/Users/maaary/Downloads/DBS-main/model.h`
- `/Users/maaary/Downloads/DBS-main/esp32_firmware/model.h`
- `/Users/maaary/Downloads/DBS-main/esp32_firmware/model_manifest.json`

When you want the FP32 comparison:

```bash
python3 scripts/prepare_esp32_benchmark.py --variant fp32
```

### 3. Compile and upload

The current helper defaults to the detected `Arduino Nano ESP32` board profile:

```bash
python3 scripts/compile_esp32_benchmark.py --upload
```

If auto-detection misses the board, pass the port explicitly:

```bash
python3 scripts/compile_esp32_benchmark.py --upload --port /dev/cu.usbmodemXXXXXXXXXXXX
```

To compile only:

```bash
python3 scripts/compile_esp32_benchmark.py
```

For the older ArduTFLite Hjorth benchmark:

```bash
python3 scripts/compile_esp32_benchmark.py --mode legacy --upload
```

### 4. Measure on-device latency

After flashing:

```bash
python3 scripts/run_esp32_benchmark.py --runs 200
```

For the older ArduTFLite Hjorth benchmark:

```bash
python3 scripts/run_esp32_benchmark.py --mode legacy --timeout 30
```

Optional logging:

```bash
python3 scripts/run_esp32_benchmark.py \
  --runs 500 \
  --save-log results/esp32/int8_serial.log \
  --output-json results/esp32/int8_bench.json
```

If you want to test a custom observation:

```bash
python3 scripts/run_esp32_benchmark.py \
  --observation 0.5 0.3 0.7 0.2 \
  --runs 200
```

## Serial Commands

At 115200 baud, the sketch accepts:

- `help`
- `info`
- `defaults`
- `sample <v1> <v2> ... <vN>`
- `run`
- `bench <count>`

The Python helper wraps `sample` and `bench` automatically.

## What The Firmware Measures

- `quantize` - Time spent converting float inputs into int8 when the model is quantized
- `invoke` - The actual TFLite Micro inference call
- `dequant` - Time spent decoding int8 outputs back to float
- `total` - End-to-end inference time inside the sketch

This separation matters because `INT8` can have a faster kernel but still lose in total time if input/output conversion dominates.

## Notes For This Repo

- The current prepared INT8 artifact is a 4D actor unless you regenerate a 6D `.tflite` model first.
- The firmware accepts both 4D and 6D observations, but the embedded model and the input you send must match.
- Desktop TFLite delegate results and ESP32 TFLite Micro results will not match exactly, because they use different runtimes and different low-level kernels.

## Troubleshooting

### Compile fails in `TensorFlowLite_ESP32`

This firmware no longer depends on that older package. Use `Chirale_TensorFLowLite` instead.

### `AllocateTensors() failed`

- Increase `kTensorArenaSize` in `/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_inference.ino`
- Make sure the generated header matches the model you think you flashed

### Model header missing or stale

Re-run:

```bash
python3 scripts/prepare_esp32_benchmark.py --variant int8
```

or:

```bash
python3 scripts/prepare_esp32_benchmark.py --variant fp32
```

### INT8 is still slower

- Compare `invoke_avg_us` before comparing `total_avg_us`
- If `invoke` is faster but `total` is slower, the extra cost is mostly quantize/dequant work
- If `invoke` is also slower, the bottleneck is the on-device runtime/kernel path, not the Python export path
- For this small actor, gains may be modest even when INT8 is working correctly
