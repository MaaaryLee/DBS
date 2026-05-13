# Quantization Research Summary

Date: 2026-04-15  
Updated: 2026-04-27  
This note covers the work completed **after 6D `96x96` emerged as the first tested model size where INT8 became faster than FP32 on the ESP32-S3 deployment path**.

## 1. Executive Summary

### Main takeaway

The cleanest deployment result right now is still the **6D `96x96` model**.

Under the native ESP32-S3 deployment stack
`ESP-IDF + esp-tflite-micro + ESP-NN`
with the project's default **32 KB data cache**:

- INT8 was **3.87x faster** than FP32 on-device for `96x96`
- INT8 reduced model file size by **69.65%**
- INT8 stayed **very close** to FP32 on a **strict held-out fidelity test**
- INT8 also stayed **very close** to FP32 on a stronger **replay-based control agreement** test

A larger `400x300` model was also tested on ESP32. INT8 still won, but the speedup was much smaller than `96x96`. Based on the results so far, this larger model appears **memory-limited**: the board spent more time fetching model weights than doing arithmetic.
---

## 2. Key Publishable Numbers

### Table 1. Main candidate: 6D `96x96`

| Metric | FP32 | INT8 | Interpretation |
| --- | ---: | ---: | --- |
| Desktop TFLite mean latency | `0.0016416 ms` | `0.0013336 ms` | INT8 was `1.23x` faster on desktop TFLite |
| ESP32-S3 invoke latency, 32 KB cache | `1685.776 us` | `435.796 us` | INT8 was `3.87x` faster on-board |
| ESP32-S3 total latency, 32 KB cache | `1688.174 us` | `442.758 us` | End-to-end model-serving also favored INT8 |
| TFLite file size | `42440` bytes | `12880` bytes | INT8 reduced size by `69.65%` |
| Params | `10178` | `10178` | Same model structure, different numeric format |
| MACs per inference | `9984` | `9984` | Same amount of neural-network math |
| Strict held-out fidelity MAE | baseline | `0.0022822` vs FP32 | Average output difference was very small |
| Strict held-out max abs diff | baseline | `0.0081592` vs FP32 | Worst observed output gap was still small |
| Strict held-out Pearson r | baseline | `0.999923`, `0.999831` | FP32 and INT8 changed almost identically |
| Replay agreement: frequency MAE | baseline | `0.1223 Hz` vs FP32 | Very small difference on replayed held-out trajectories |
| Replay agreement: amplitude MAE | baseline | `8.1046` vs FP32 | Small relative to the `0..5000` amplitude scale |

Trace files:
- [bench_fp32_96_96_repaired.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/bench_fp32_96_96_repaired.json)
- [bench_int8_96_96_repaired.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/bench_int8_96_96_repaired.json)
- [native_fp32_96_96_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_summary.json)
- [native_int8_96_96_repaired_32k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_repaired_32k_summary.json)
- [eval_tflite_96_96_strict_holdout.json](/Users/maaary/Downloads/DBS-main/results/larger_models/96_96/eval_tflite_96_96_strict_holdout.json)

### Table 2. Size sweep on native ESP32-S3, 32 KB cache

| Model | Params | MACs | FP32 invoke (`us`) | INT8 invoke (`us`) | INT8 speedup | What we learned |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `64x32` | smaller | smaller | `124.28` | `165.22` | `0.75x` | FP32 still faster |
| `64x64` | larger | larger | `196.38` | `246.10` | `0.80x` | FP32 still faster |
| `80x80` | `7202` | about `7040` | `281.908` | `333.686` | `0.845x` | Still below crossover |
| `96x96` | `10178` | `9984` | `1685.776` | `435.796` | `3.87x` | First tested INT8 winner |
| `128x128` | `17666` | larger | `2885.166` | `680.562` | `4.24x` | INT8 still wins |
| `400x300` | `123702` | `123000` | `19799.883` | `10505.610` | `1.88x` | INT8 still wins, but much less strongly |

Trace files:
- [summary_6d_models.json](/Users/maaary/Downloads/DBS-main/results/larger_models/summary_6d_models.json)
- [native_fp32_400_300_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_400_300_summary.json)
- [native_int8_400_300_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_400_300_summary.json)
- [pipeline_manifest_400_300.json](/Users/maaary/Downloads/DBS-main/results/larger_models/400_300/pipeline_manifest_400_300.json)

### Table 3. Cache sensitivity checks

| Model | Cache setting | FP32 invoke (`us`) | INT8 invoke (`us`) | Conclusion |
| --- | --- | ---: | ---: | --- |
| `96x96` | `32 KB` | `1685.776` | `435.796` | INT8 clearly faster |
| `96x96` | `64 KB` | `375.790` | `435.833` | FP32 became faster once it got more cache |
| `400x300` | `32 KB` | `19799.883` | `10505.610` | INT8 faster |
| `400x300` | `64 KB` | `19808.557` | `10502.503` | INT8 still faster by about `1.89x`; bigger cache alone still did not materially change the result |

Trace files:
- [native_fp32_96_96_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_dc64k_summary.json)
- [native_int8_96_96_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_dc64k_summary.json)
- [native_fp32_400_300_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_400_300_dc64k_summary.json)
- [native_int8_400_300_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_400_300_dc64k_summary.json)
- [native_int8_400_300_dc64k_verify.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_400_300_dc64k_verify.json)

---

## 3. Research Process by Problem Category

The sections below follow the same pattern:

**Problem -> Question/confusion -> Method -> Result -> Limitation / reflection**

The categories are ranked roughly by how important they seem for the final paper story.

---

## 4. Problem Category 1: Inference Time on the Real Board

This was the biggest scientific problem to sort out.

### Problem

At first, INT8 was **not** consistently faster than FP32 on ESP32. That was surprising, because quantization is often introduced as something that should make inference faster.

### Question / confusion

The main questions were:

- Why is INT8 still slower?
- Is the INT8 kernel missing?
- Is the problem caused by the firmware path, not the model?
- Is the model too small for INT8 to help?

### Method

The comparison was cleaned up step by step:

1. We separated desktop benchmarking from on-board benchmarking.
2. The workflow moved from the older Arduino-style path to the **native ESP32 path**:
   - `ESP-IDF + esp-tflite-micro + ESP-NN`
3. The native path was confirmed to be using Espressif's optimized INT8 path.
4. A **6D-only size sweep** was run so the legacy 4D models would stop mixing with the 6D deployment story.
5. FP32 vs INT8 was benchmarked on the same board, same runtime family, same model structure.

### Result

The result showed a clear threshold behavior.

- `64x32`, `64x64`, and `80x80`: FP32 still won
- `96x96`: INT8 became the first tested winner
- `128x128`: INT8 also won

So the practical finding was:

> In this deployment stack, `96x96` was the first tested 6D model size where INT8 clearly beat FP32 on the ESP32-S3.

### Why this matters

This produced a main candidate model that was both:

- small enough to deploy comfortably
- large enough for INT8 acceleration to actually pay off

### Limitation / reflection

This threshold is not a universal law.

It depends on:

- this runtime path
- this board
- this cache setting
- this model family

So the most accurate statement is:

> `96x96` was the first tested crossover point in **this project's deployment setup**.

---

## 5. Problem Category 2: Cache Dependence and Why `96x96` Looked So Good

This was the second biggest issue, because it changed how strongly the paper claim could be phrased.

### Problem

`96x96` gave a very large INT8 speedup at `32 KB` cache, but we needed to know whether that was a deep model property or a memory-system effect.

### Question / confusion

The main confusion was:

- Is `96x96` just a better model?
- Or is the board's memory/cache configuration helping INT8 more than FP32?

### Method

A cache ablation was run on the native ESP32-S3 path.

For `96x96`, we compared:

- default `32 KB` data cache
- increased `64 KB` data cache

Later, the `400x300` native benchmark was also switched to `64 KB` cache and verified from the board output to really report `data_cache_kb=64` on boot.

### Result

For `96x96`:

- at `32 KB`: FP32 `1685.776 us`, INT8 `435.796 us`
- at `64 KB`: FP32 `375.790 us`, INT8 `435.833 us`

This showed that:

- INT8 hardly changed
- FP32 improved dramatically when it got more cache

Interpretation:

> The `96x96` INT8 model was already small enough that it fit the board's fast memory situation pretty well. FP32 was suffering much more from limited cache. Once FP32 got a larger cache, its latency dropped sharply and the advantage disappeared.

For `400x300` INT8:

- at `32 KB`: `10505.610 us`
- at `64 KB`: `10502.520 us`

So the larger model barely changed on the INT8 side.

### Why this matters

This means the `96x96` result is a **deployment-specific win**, not a general hardware truth.

### Limitation / reflection

At the time this section was first written, there was **not** yet a full matched `400x300 FP32 vs INT8 @ 64 KB` pair. Only the `400x300` INT8 side at `64 KB` was available.

The matched `400x300 FP32 vs INT8 @ 64 KB` comparison has now been added, and it tells the same basic story: the speedup stayed about the same, so simply moving from `32 KB` to `64 KB` did not materially change the large-model result.

---

## 6. Problem Category 3: Why the Very Large `400x300` Model Did Not Win by More

This became important because `400x300` is the paper-inspired larger model that matters for the replication discussion.

### Problem

A much bigger model might have been expected to make INT8 win even more strongly. But that did **not** happen.

### Question / confusion

The main confusion was:

- If INT8 is good, shouldn't the speedup keep growing as the model gets bigger?
- Why was `400x300` only `1.88x` faster, while `96x96` was `3.87x` faster?

### Method

A full repo-native **6D `400x300` model family** was trained, exported, and benchmarked on:

- desktop TFLite
- native ESP32-S3 runtime

The native cache was also increased to `64 KB`, and the comparison was re-run.

### Result

For desktop TFLite:

- FP32 `0.0093839 ms`
- INT8 `0.0032820 ms`
- INT8 speedup `2.86x`

For ESP32-S3, `32 KB` cache:

- FP32 `19799.883 us`
- INT8 `10505.610 us`
- INT8 speedup `1.88x`

For ESP32-S3, `64 KB` cache, INT8 only:

- INT8 `10502.520 us`
- almost unchanged from `32 KB`

### Interpretation

The most likely explanation from these measurements is:

> The large model became more **memory-limited** than **compute-limited**.

That means the board spent a lot of time **fetching weights** rather than just doing arithmetic.

The practical intuition is:

- `96x96` is small enough that INT8 benefits strongly from the board's small fast memory
- `400x300` is so large that both FP32 and INT8 have to keep streaming lots of weights
- INT8 still helps, but the relative gain becomes smaller

### Limitation / reflection

This explanation is strongly supported by the measurements, but it is still an interpretation of the system behavior, not a formal hardware proof.

The next best check was a **fully matched `400x300 FP32 vs INT8 @ 64 KB` pair**, and that pair has now been run. The result was still about `1.89x` in favor of INT8, so increasing cache from `32 KB` to `64 KB` did not rescue the large model.

---

## 7. Problem Category 4: Evidence Validity and Benchmark Trustworthiness

This became the main "can we trust the whole story?" section.

### Problem

Several parts of the evidence chain were too loose at first:

- desktop TFLite and ESP32 did not behave the same way
- the original 6D evaluation state file collapsed to effectively one repeated state
- the early control check was deterministic and too close to a smoke test
- multiple runtime paths made it too easy to compare the wrong model or wrong benchmark path

### Question / confusion

The main trustworthiness questions were:

- Which benchmark is the real deployment benchmark?
- Are the fidelity numbers based on truly unseen states?
- Does INT8 preserve control behavior beyond a smoke test?
- Is the firmware definitely running the intended model, runtime, and cache setting?

### Method

This was tightened in four steps:

1. Desktop TFLite was separated from native ESP32 and treated as supporting context rather than deployment proof.
2. The 6D state generation was repaired, degenerate sets were rejected, and a strict calibration / held-out split was built with `500` source indices per side and `0` overlap.
3. A replay-based control agreement test was added on `50` held-out episodes with `10` steps each, for `500` held-out states total.
4. The main benchmark path was unified around **6D-only native ESP-IDF**, with explicit runtime metadata such as `model_io`, `input_dim`, `output_dim`, and `data_cache_kb` printed by the board.

Trace files:
- [states_calibration_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_calibration_6d_strict_metadata.json)
- [states_heldout_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_heldout_6d_strict_metadata.json)
- [replay_episodes_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/replay_episodes_6d_strict_metadata.json)
- [native_int8_400_300_dc64k_verify.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_400_300_dc64k_verify.json)

### Result

The evidence chain is now much stronger:

- **Desktop vs board role split**
  desktop TFLite stayed useful for software-side trend checking, but ESP32 remained the deployment-relevant benchmark
- **Strict held-out fidelity for `96x96`**
  unique held-out rows `499`, overlap `0`, MAE `0.0022822`, max abs diff `0.0081592`, Pearson `0.999923` and `0.999831`
- **Replay-based control agreement for `96x96`**
  frequency MAE `0.1223 Hz`, amplitude MAE `8.1046`
- **Firmware/runtime verification**
  the board now reports enough metadata directly to verify the active runtime path and cache setting from the trace output

### Limitation / reflection

- replay-based agreement is still not the same as fresh stochastic closed-loop simulation
- desktop TFLite remains supporting context, not the main deployment proof
- Arduino is still useful for manual checks and power measurement, but native ESP-IDF remains the cleaner latency benchmark

---

## 8. Problem Category 5: Arduino `400x300` Rerun with Cache-Blocking 

This follow-up became necessary once the large-model bottleneck story needed a fresh same-board rerun and a clearer systems debugging trail.

### Problem

After the standalone cache-blocking study suggested that `400x300` was strongly memory-limited, there was still no fresh same-board rerun from the currently attached Arduino Nano ESP32 documenting the full problem-solving path. The board would enumerate over USB, but upload handshakes were unstable, and the Arduino benchmark sketch could not hold a large tensor arena in internal DRAM.

### Question / confusion

The main practical questions were:

- Was the board unavailable, or just switching between runtime USB mode and bootloader USB mode?
- Could the Arduino benchmark path hold the `400x300` model at all, or was the tensor arena too large for internal DRAM?
- If cache blocking helped an isolated dense kernel by more than `2x`, would the real board-level `400x300` result still look memory-limited?

### Method

The debugging and measurement workflow on April 27, 2026 was:

1. The Arduino Nano ESP32 was reprobed until both its normal serial endpoint and its bootloader serial endpoint were understood. Uploads started from the normal runtime port `/dev/cu.usbmodem3C8427C3CECC2`, but successful flashing often required finishing on the ESP32-S3 bootloader port `/dev/cu.usbmodem101` once the device re-enumerated.
2. Fresh Arduino benchmark artifacts were generated for both `400x300 INT8` and `400x300 FP32`.
3. The larger sketch initially failed to link because a static `256 KB` tensor arena overflowed `.dram0.bss` by `16824` bytes. To make the larger model runnable, the tensor arena was moved out of internal DRAM and into PSRAM by allocating it at runtime with `heap_caps_aligned_alloc(..., MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)` in [dbs_inference.ino](/Users/maaary/Downloads/DBS-main/esp32_firmware/dbs_inference.ino).
4. After that change, both `400x300` sketches compiled, flashed, and were benchmarked with `3` repeated runs of `200` inferences each on the same connected board.
5. In parallel, the standalone cache-blocking study was kept separate and reported only as a dense-kernel systems experiment, not as an on-board TFLite Micro speedup.

### Result

Fresh Arduino `400x300` board rerun:

- INT8 invoke `9341.950 us`, total `9349.370 us`
- FP32 invoke `19320.837 us`, total `19323.813 us`
- INT8 speedup `2.07x` on invoke and `2.07x` on total

Saved traces:

- [arduino_int8_400_300_retry_20260427_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/arduino_int8_400_300_retry_20260427_summary.json)
- [arduino_fp32_400_300_retry_20260427_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/arduino_fp32_400_300_retry_20260427_summary.json)

Standalone cache-blocking `400x300` follow-up:

- best packed tiling seen in this sweep: naive `94637.310 ns` to blocked packed `33255.642 ns`, which is `2.85x`
- strongest row-major blocking seen in this sweep: naive `92469.136 ns` to blocked row-major `36631.062 ns`, which is `2.52x`

Saved traces:

- [cache_blocking_400x300_rt32_ct32_20260427_manual.json](/Users/maaary/Downloads/DBS-main/results/cache_blocking/cache_blocking_400x300_rt32_ct32_20260427_manual.json)
- [cache_blocking_400x300_rt8_ct16_20260427_manual.json](/Users/maaary/Downloads/DBS-main/results/cache_blocking/cache_blocking_400x300_rt8_ct16_20260427_manual.json)

### Why this matters

This follow-up strengthens the same basic systems diagnosis:

- the live board rerun still showed only a moderate `400x300` INT8 win, which is consistent with a memory-limited model
- the isolated dense-kernel study showed that better weight layout and tiling can plausibly unlock larger gains if a custom kernel is ever integrated into firmware

So the new evidence does not overturn the earlier story. It makes the bottleneck explanation more concrete.

### Limitation / reflection

- These fresh April 27, 2026 results came from the Arduino benchmark sketch path, not the native ESP-IDF benchmark path used for the main manuscript claim.
- The PSRAM-backed tensor arena was a practical way to make `400x300` runnable in the Arduino path, but it also means these absolute latencies should be treated as supporting systems evidence, not a replacement for the native ESP-IDF tables above.
- The cache-blocking speedups are not yet on-board TFLite Micro speedups. They explain the likely bottleneck and suggest an optimization direction, but they should not be multiplied into the board-level FP32 vs INT8 result until that kernel work is actually integrated into firmware.

---

## 9. Summary

### Strong enough to show today

- The main deployment candidate is **6D `96x96` INT8**.
- On native ESP32-S3 at `32 KB` cache, INT8 beat FP32 by `3.87x`.
- INT8 reduced TFLite model size by `69.65%`.
- INT8 stayed very close to FP32 on a strict held-out fidelity test.
- INT8 also stayed very close to FP32 on replayed held-out trajectories.
- The larger `400x300` model also showed an INT8 speedup on ESP32, but the speedup dropped to `1.88x`.
- A fresh Arduino rerun on April 27, 2026 kept the `400x300` conclusion intact: INT8 still beat FP32 by about `2.07x` on the connected board.
- A standalone `400x300` cache-blocking sweep showed `2.52x` to `2.85x` dense-kernel speedups, which supports the interpretation that this large model is memory-limited.
- Cache configuration matters a lot for interpretation, so the `96x96` result should be framed as a deployment-specific win, not a universal rule.

### Still missing / must be framed carefully

- A matched **FP32 vs INT8 on-board power / energy measurement** under the same benchmark conditions.
- A fresh **closed-loop stochastic control** evaluation, not just replay agreement.
- If strict replication of the prior paper is required, a careful methods comparison against that paper's exact runtime and memory configuration.
- Desktop TFLite should stay as supporting context, not a substitute for on-board deployment measurement.
- The April 27, 2026 Arduino reruns should be treated as supporting systems checks, not replacements for the native ESP-IDF benchmark tables above.
- The cache-blocking speedups are still standalone kernel results; they are not yet an integrated firmware-side optimization.

### Questions for advisor:

1. Should the paper's main result center on the deployable `96x96` controller, or on closer replication of the larger `400x300` reference architecture?
2. Before moving into pruning or compression, should the next systems step focus on memory-side optimization for `400x300`, such as cache blocking or weight-layout work?
3. Is the current replay-based evidence enough for this milestone, or is a fresh closed-loop stochastic test needed before submission?

### One-paragraph version

On the native ESP32-S3 deployment path, `96x96` was the first tested 6D model where INT8 clearly became faster than FP32. Under the default `32 KB` data-cache configuration, INT8 reduced on-device invoke latency from `1685.776 us` to `435.796 us` and reduced model size from `42440` to `12880` bytes. The 6D evaluation pipeline was then repaired, a strict calibration/held-out split with zero overlap was created, and INT8 stayed very close to FP32 on unseen states and replayed held-out trajectories. A much larger `400x300` model inspired by the prior paper was also tested. INT8 still won on ESP32, but only by `1.88x`, suggesting the model became more memory-limited. The strongest current claim is therefore deployment-specific: `96x96 INT8` is the cleanest current candidate for this ESP32-S3 setup, while `400x300` is better framed as a larger-capacity model that exposes memory-system limits.
