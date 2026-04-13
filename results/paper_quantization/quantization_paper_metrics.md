# Quantization Evidence Bundle

Generated: 2026-04-13T04:13:29.463298+00:00

## Candidate 96x96 (6D)

- Params: 10178
- MACs per inference: 9984
- FP32 parameter bytes: 40712
- INT8 parameter bytes: 10178
- Model size reduction: 69.65%
- Desktop TFLite latency: FP32 0.001641600 ms, INT8 0.001333610 ms, INT8 speedup 1.231x
- ESP32 default-cache latency: FP32 1685.776 us, INT8 435.796 us, INT8 speedup 3.868x
- ESP32 64 KB cache latency: FP32 375.790 us, INT8 435.833 us
- Repaired state coverage: 987 unique states
- Fidelity MAE: 0.002338831
- Fidelity max abs diff: 0.008183837

## Threshold Screening

| Model | Params | FP32 bytes | INT8 bytes | ESP32 FP32 us | ESP32 INT8 us | INT8 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 80_80 | 7202 | 28808 | 7202 | 281.908 | 333.686 | 0.845x |
| 96_96 | 10178 | 40712 | 10178 | 1685.776 | 435.796 | 3.868x |
| 128_128 | 17666 | 70664 | 17666 | 2885.166 | 680.562 | 4.239x |

## Trace Files

- Candidate desktop FP32: `/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/bench_fp32_96_96_repaired.json`
- Candidate desktop INT8: `/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/bench_int8_96_96_repaired.json`
- Candidate ESP32 FP32: `/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_summary.json`
- Candidate ESP32 INT8: `/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json`
- Candidate fidelity: `/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/eval_tflite_96_96_repaired.json`
- Original 6D states: `/Users/maaary/Downloads/DBS-main/states_eval_6d.npy`
- Repaired 6D states: `/Users/maaary/Downloads/DBS-main/states_eval_6d_repaired.npy`

## Things That Need Caution

- [high/fixed for the candidate 96x96 rerun] The original 6D representative dataset collapsed to one repeated state. Impact: It invalidated the original 6D fidelity/correlation analysis and likely distorted INT8 calibration ranges.
- [high/open limitation] The cached 6D environment is deterministic in offline mode. Impact: The env_eval block is only a functional smoke test; it is not strong evidence for stochastic control performance.
- [medium/open limitation] The repaired 6D fidelity evaluation uses the same state-set family that was also used for INT8 calibration. Impact: The fidelity numbers are useful and quantifiable, but they should be described as in-distribution or calibration-family fidelity unless a held-out 6D state set is added.
- [high/open nuance] The 96x96 INT8 speedup on ESP32-S3 depends on the default 32 KB data-cache configuration. Impact: The paper must not claim a hardware-wide INT8 win without stating the cache/runtime configuration.
- [medium/updated after repaired rerun] The 80x80 and 128x128 threshold-screening models were rerun with repaired INT8 calibration, but they still only have latency-screening traces rather than full repaired fidelity/control evaluations. Impact: The repaired threshold crossover is valid as a latency result, but only the 96x96 candidate currently has the full repaired fidelity/control evidence bundle.
- [medium/fixed] The desktop TFLite latency helpers could silently append another `_6d` suffix and benchmark a synthesized fallback file. Impact: The repaired desktop traces are valid; older traces should be checked if they relied on custom state-file names.

## Source Links

- Arduino Nano ESP32: https://docs.arduino.cc/hardware/nano-esp32
- esp-tflite-micro: https://github.com/espressif/esp-tflite-micro
- ESP-NN: https://github.com/espressif/esp-nn
- TensorFlow Lite integer quantization: https://www.tensorflow.org/lite/performance/post_training_integer_quant
