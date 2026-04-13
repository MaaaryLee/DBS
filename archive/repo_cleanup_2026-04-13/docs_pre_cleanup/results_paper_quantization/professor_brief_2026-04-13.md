# Professor Brief

Date: 2026-04-13

This is the clean snapshot of what is true today, what evidence backs it up, and what is still missing before the work is fully paper-ready.

## One-Minute Summary

- We found the first tested 6D model size where INT8 beats FP32 on the ESP32-S3 deployment path: `96x96`.
- Under the current ESP32-S3 deployment configuration, `96x96` INT8 is about `3.87x` faster than FP32 on-device.
- We now also have a strict disjoint calibration-vs-held-out fidelity test.
- On that strict held-out test, INT8 remains very close to FP32.
- We also added a stronger replay-based control agreement test over many varied held-out trajectories.
- The main remaining gap is that the strongest control claim would still require fresh closed-loop MATLAB dynamics, not only replayed held-out observations.

## Main Claim We Can Safely Make Today

Under the current deployment configuration for Arduino Nano ESP32 / ESP32-S3 (`ESP-IDF + esp-tflite-micro + ESP-NN`, default `32 KB` data cache), the 6D `96x96` INT8 policy reduces on-device inference latency and model size relative to FP32 while maintaining high agreement with FP32 on a strict held-out state split and on replayed held-out trajectories.

## What We Have

### 1. On-device ESP32 latency result

Primary file:
- [quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)

Trace files:
- [native_fp32_96_96_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_summary.json)
- [native_int8_96_96_repaired_32k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json)

Current headline numbers:
- FP32 `invoke_avg_us = 1685.776`
- INT8 `invoke_avg_us = 435.796`
- INT8 speedup vs FP32: `3.868x`

Important caution:
- These on-device numbers currently correspond to the repaired `96x96` INT8 model used in the existing paper bundle.
- We have not yet rerun the ESP32 latency measurement with the newly strict-calibrated `model_int8_96_96_strict.tflite`.

### 2. Cache-ablation result

Trace files:
- [native_fp32_96_96_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_dc64k_summary.json)
- [native_int8_96_96_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_dc64k_summary.json)

Current numbers:
- `32 KB` cache: FP32 `1685.776 us`, INT8 `435.796 us`
- `64 KB` cache: FP32 `375.79 us`, INT8 `435.833 us`

Plain meaning:
- INT8 wins in the current deployment configuration.
- If FP32 gets a larger cache, FP32 becomes faster again.
- So this is a deployment-specific win, not a universal hardware-wide win.

### 3. Strict held-out fidelity test

Strict split artifacts:
- [states_calibration_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_calibration_6d_strict_metadata.json)
- [states_heldout_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_heldout_6d_strict_metadata.json)

Key validity fact:
- Calibration/held-out overlap: `0`

Why that matters:
- Calibration data is used to tune INT8 ranges.
- Held-out data is the unseen test.
- This avoids “testing on the same examples used to prepare the quantized model.”

Primary strict result:
- [eval_tflite_96_96_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/eval_tflite_96_96_strict_holdout.json)

Key strict `96x96` numbers:
- Held-out MAE: `0.002282`
- Held-out max abs diff: `0.008159`
- Pearson: `0.999923`, `0.999831`
- Input saturation: `0.000333`

Plain meaning:
- INT8 and FP32 remain extremely close even on unseen 6D states.

### 4. Stronger replay-based control agreement

Replay artifact:
- [replay_episodes_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/replay_episodes_6d_strict_metadata.json)

Replay setup:
- `50` episodes
- `10` steps per episode
- `500` held-out states total

Primary strict replay result:
- [eval_tflite_96_96_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/eval_tflite_96_96_strict_holdout.json)

Key replay `96x96` agreement numbers:
- Action MAE: `0.002282`
- Frequency MAE: `0.122 Hz`
- Amplitude MAE: `8.10` on a `0..5000` amplitude scale

Plain meaning:
- Across many varied unseen trajectories, INT8 chooses nearly the same DBS commands as FP32.

Important caution:
- This is stronger than the old deterministic smoke test.
- It is still not the same as fresh closed-loop MATLAB simulation.

### 5. Strictly evaluated ablation models

Files:
- [eval_tflite_80_80_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/80_80/eval_tflite_80_80_strict_holdout.json)
- [eval_tflite_128_128_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/128_128/eval_tflite_128_128_strict_holdout.json)

Summary:
- `80x80`: held-out MAE `0.002367`
- `96x96`: held-out MAE `0.002282`
- `128x128`: held-out MAE `0.002520`

Plain meaning:
- All three strict INT8 models remain very close to FP32.
- So the main distinction between these sizes is latency/deployment behavior, not obvious quantization collapse.

## What Is Still Missing

### 1. Strict-calibrated on-device ESP32 rerun

What is missing:
- We have strict held-out fidelity and replay results for the new strict-calibrated INT8 models.
- But the current on-device ESP32 speedup still comes from the earlier repaired `96x96` INT8 model, not the new strict-calibrated one.

Why it matters:
- For the cleanest story, the on-device deployment result and the held-out fidelity result should reference the same final INT8 artifact.

### 2. Fresh closed-loop stochastic control evidence

What is missing:
- The new replay test uses many held-out trajectories, but it still replays cached signals.
- It does not yet simulate fresh dynamics that respond online to the controller’s actions.

Why it matters:
- Replay agreement shows that INT8 and FP32 behave similarly on varied unseen inputs.
- Fresh closed-loop evaluation would be the stronger evidence that the controller remains valid in a more realistic control loop.

### 3. Updated paper bundle

What is missing:
- [quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md) still centers the repaired pre-strict fidelity result.
- The new strict held-out and replay evidence has not yet been folded into the main machine-generated paper bundle.

## Best Questions To Ask Professor Today

1. Is the current scope acceptable as a deployment-focused quantization paper if we clearly state that the strongest control evidence is strict held-out fidelity plus replay agreement, not fresh closed-loop MATLAB evaluation yet?
2. Does he want the paper to center only the `96x96` candidate, with `80x80` and `128x128` kept as ablation evidence?
3. Is rerunning ESP32 latency with the strict-calibrated `96x96` model sufficient to unify the final deployment and fidelity story?
4. Does he consider replay-based held-out agreement strong enough for this stage, or does he want fresh MATLAB closed-loop evidence before submission?
5. How narrowly should the main claim be written regarding the cache/runtime dependency?

## Plain-Language Explanations To Use In Meeting

### Why calibration and held-out must be disjoint

If the quantized model is “prepared” using one set of examples, and then we evaluate it on those same examples, the test is too easy. A disjoint held-out test is like a real exam: the model has to match FP32 on inputs it did not already see during quantization setup.

### What cache/runtime dependency means

INT8 is faster here because the model is smaller and fits the current deployment memory setup better. But if we give FP32 a larger cache, FP32 speeds up a lot. So the result depends on the exact firmware/runtime configuration, not just on “INT8 vs FP32” in the abstract.

### Deployment-specific win vs general hardware win

Safe:
- “INT8 is faster in our current ESP32-S3 deployment setup.”

Not safe:
- “INT8 is always faster on ESP32-S3.”

## Files To Open Live If Needed

Start with:
- [professor_brief_2026-04-13.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/professor_brief_2026-04-13.md)

Then show:
- [quantization_paper_metrics.md](/Users/maaary/Downloads/DBS-main/results/paper_quantization/quantization_paper_metrics.md)
- [eval_tflite_96_96_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/eval_tflite_96_96_strict_holdout.json)
- [states_calibration_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_calibration_6d_strict_metadata.json)
- [states_heldout_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_heldout_6d_strict_metadata.json)
- [native_fp32_96_96_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_summary.json)
- [native_int8_96_96_repaired_32k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json)
