# Quantization Research Summary for PI

Date: 2026-04-15  
Project scope in this summary: everything we did **after we discovered that 6D `96x96` was the first tested model size where INT8 became faster than FP32 on the ESP32-S3 deployment path**.

## 1. Executive Summary

### Main takeaway

The cleanest deployment result in this project is still the **6D `96x96` model**.

Under the native ESP32-S3 deployment stack
`ESP-IDF + esp-tflite-micro + ESP-NN`
with the project's default **32 KB data cache**:

- INT8 was **3.87x faster** than FP32 on-device for `96x96`
- INT8 reduced model file size by **69.65%**
- INT8 stayed **very close** to FP32 on a **strict held-out fidelity test**
- INT8 also stayed **very close** to FP32 on a stronger **replay-based control agreement** test

The larger `400x300` model also showed an INT8 speedup on ESP32, but the speedup was much smaller than `96x96`. The current evidence suggests that this larger model became **memory-limited**: the board spent more time fetching model weights than doing arithmetic.

### Best publishable claim right now

> Under the native ESP32-S3 deployment configuration used in this project (`ESP-IDF + esp-tflite-micro + ESP-NN`, default `32 KB` data cache), the 6D `96x96` INT8 policy reduced on-device inference latency and model size relative to FP32 while maintaining high agreement with FP32 on a strict held-out state split and replayed held-out trajectories.

### What not to overclaim

We should **not** say:

- "INT8 is always faster on ESP32-S3"
- "INT8 is always faster for all model sizes"
- "Desktop TFLite speedups prove board deployment speedups"

The data supports a **deployment-specific** claim, not a universal one.

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
| `400x300` | `64 KB` | not yet rerun | `10502.520` | INT8 hardly changed; bigger cache alone did not noticeably improve INT8 |

Trace files:
- [native_fp32_96_96_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_fp32_96_96_dc64k_summary.json)
- [native_int8_96_96_dc64k_summary.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_96_96_dc64k_summary.json)
- [native_int8_400_300_dc64k_verify.json](/Users/maaary/Downloads/DBS-main/results/esp32/repeats/native_int8_400_300_dc64k_verify.json)

---

## 3. Research Process by Problem Category

The sections below follow the same pattern:

**Problem -> Question/confusion -> Method -> Result -> Limitation / reflection**

The categories are ranked roughly by how important they were to the final paper story.

---

## 4. Problem Category 1: Inference Time on the Real Board

This was the biggest scientific problem.

### Problem

At first, INT8 was **not** consistently faster than FP32 on ESP32. That was surprising, because quantization is often introduced as something that should make inference faster.

### Question / confusion

The main questions were:

- Why is INT8 still slower?
- Is the INT8 kernel missing?
- Is the problem caused by the firmware path, not the model?
- Is the model too small for INT8 to help?

### Method

We cleaned up the comparison step by step:

1. We separated desktop benchmarking from on-board benchmarking.
2. We moved from the older Arduino-style path to the **native ESP32 path**:
   - `ESP-IDF + esp-tflite-micro + ESP-NN`
3. We confirmed that the native path really was using Espressif's optimized INT8 path.
4. We ran a **6D-only size sweep** so we would stop mixing legacy 4D models with the 6D deployment story.
5. We benchmarked FP32 vs INT8 on the same board, same runtime family, same model structure.

### Result

The size sweep showed a clear threshold behavior.

- `64x32`, `64x64`, and `80x80`: FP32 still won
- `96x96`: INT8 became the first tested winner
- `128x128`: INT8 also won

So the practical finding was:

> In this deployment stack, `96x96` was the first tested 6D model size where INT8 clearly beat FP32 on the ESP32-S3.

### Why this matters

This gave us a main candidate model that was both:

- small enough to deploy comfortably
- large enough for INT8 acceleration to actually pay off

### Limitation / reflection

This threshold is **not a universal law**.

It depends on:

- this runtime path
- this board
- this cache setting
- this model family

So the right statement is:

> `96x96` was the first tested crossover point in **this project's deployment setup**.

---

## 5. Problem Category 2: Cache Dependence and Why `96x96` Looked So Good

This was the second biggest issue, because it changed how strongly we could phrase the paper claim.

### Problem

`96x96` gave a very large INT8 speedup at `32 KB` cache, but we needed to know whether that was a deep model property or a memory-system effect.

### Question / confusion

The main confusion was:

- Is `96x96` just a better model?
- Or is the board's memory/cache configuration helping INT8 more than FP32?

### Method

We ran a cache ablation on the native ESP32-S3 path.

For `96x96`, we compared:

- default `32 KB` data cache
- increased `64 KB` data cache

Later, we also switched the `400x300` native benchmark to `64 KB` cache and verified that the board really reported `data_cache_kb=64` on boot.

### Result

For `96x96`:

- at `32 KB`: FP32 `1685.776 us`, INT8 `435.796 us`
- at `64 KB`: FP32 `375.790 us`, INT8 `435.833 us`

This means:

- INT8 hardly changed
- FP32 improved dramatically when it got more cache

Plain-language interpretation:

> The `96x96` INT8 model was already small enough that it fit the board's fast memory situation pretty well. FP32 was suffering much more from limited cache. Once FP32 got a larger cache, its latency dropped sharply and the advantage disappeared.

For `400x300` INT8:

- at `32 KB`: `10505.610 us`
- at `64 KB`: `10502.520 us`

So the larger model barely changed on the INT8 side.

### Why this matters

This tells us that the `96x96` result is a **deployment-specific win**, not a general hardware truth.

### Limitation / reflection

We still do **not** have a full matched `400x300 FP32 vs INT8 @ 64 KB` pair. Right now we only verified the `400x300` INT8 side at `64 KB`.

So the strongest cache conclusion remains the `96x96` pair.

---

## 6. Problem Category 3: Why the Very Large `400x300` Model Did Not Win by More

This became important because `400x300` is the paper-inspired larger model that your PI cares about.

### Problem

We expected that a much bigger model might make INT8 win even more strongly. But that did **not** happen.

### Question / confusion

The confusion was:

- If INT8 is good, shouldn't the speedup keep growing as the model gets bigger?
- Why was `400x300` only `1.88x` faster, while `96x96` was `3.87x` faster?

### Method

We trained and exported a full repo-native **6D `400x300` model family** and benchmarked it on:

- desktop TFLite
- native ESP32-S3 runtime

We also increased the native cache to `64 KB` and re-verified the INT8 run.

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

### Plain-language interpretation

The most likely explanation is:

> The large model became more **memory-limited** than **compute-limited**.

That means the board spent a lot of time **fetching weights** rather than just doing arithmetic.

The practical intuition is:

- `96x96` is small enough that INT8 benefits strongly from the board's small fast memory
- `400x300` is so large that both FP32 and INT8 have to keep streaming lots of weights
- INT8 still helps, but the relative gain becomes smaller

### Limitation / reflection

This explanation is strongly supported by the measurements, but it is still an interpretation of the system behavior, not a formal hardware proof.

The next strongest systems check would be a **fully matched `400x300 FP32 vs INT8 @ 64 KB` pair**.

---

## 7. Problem Category 4: Desktop TFLite vs ESP32 Results Did Not Match

This was an important interpretation problem.

### Problem

Desktop TFLite often showed smaller but smoother INT8 wins, while the board showed behavior that depended much more on model size and cache.

### Question / confusion

The confusion was:

- Which result is more "scientific"?
- Can desktop TFLite stand in for the board?
- Why does the board behave so differently?

### Method

We explicitly separated the two roles:

- **Desktop TFLite** = software-side benchmark
- **ESP32 benchmark** = deployment benchmark

We then compared the same model families across both environments.

### Result

For `96x96`:

- desktop TFLite INT8 speedup: `1.23x`
- native ESP32 INT8 speedup at `32 KB`: `3.87x`

For `400x300`:

- desktop TFLite INT8 speedup: `2.86x`
- native ESP32 INT8 speedup at `32 KB`: `1.88x`

### Plain-language interpretation

Desktop TFLite is running on a much stronger computer with:

- larger CPU caches
- more memory bandwidth
- a more mature optimized backend

The ESP32 is a small microcontroller and is much closer to the real intended deployment.

So the correct interpretation is:

> Desktop TFLite is useful for software-side trend checking, but ESP32 is the deployment-relevant benchmark.

### Limitation / reflection

Desktop numbers are still valuable, but they should be presented as **supporting context**, not as the main proof of deployable performance.

---

## 8. Problem Category 5: Fidelity Numbers Were Not Originally Trustworthy Enough

This was a major validity problem in the evaluation pipeline.

### Problem

The original 6D representative/evaluation state file collapsed to effectively one repeated state. That made the original fidelity story unsafe.

### Question / confusion

The core concern was:

- Are we accidentally testing INT8 on the same small or degenerate state set it was calibrated on?
- Are the good-looking fidelity numbers real, or just an artifact of bad evaluation data?

### Method

We fixed this in stages:

1. Repaired the 6D state generation so it produced varied states rather than one repeated row.
2. Added checks to reject degenerate 6D calibration sets.
3. Built a **strict calibration / held-out split**.
4. Verified that calibration and held-out sets had:
   - `500` source indices each
   - `0` overlap

Trace files:
- [states_calibration_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_calibration_6d_strict_metadata.json)
- [states_heldout_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/states_heldout_6d_strict_metadata.json)

### Result

For the strict held-out `96x96` evaluation:

- held-out unique rows: `499`
- calibration / held-out overlap: `0`
- MAE: `0.0022822`
- max abs diff: `0.0081592`
- Pearson r: `0.999923`, `0.999831`
- input saturation fraction: `0.0003333` (`1` clipped input value out of `3000` total input values)

### Plain-language interpretation

This means:

- INT8 stayed very close to FP32 even on **unseen** inputs
- the agreement was not just because we tested on the same calibration rows
- the quantized model did not obviously break policy behavior

### Limitation / reflection

This is now a real held-out fidelity test, but it is still a **state-based comparison**, not a fully interactive closed-loop test.

---

## 9. Problem Category 6: Control Evaluation Was Too Weak at First

This was the main evidence-quality problem after fidelity was repaired.

### Problem

The earlier control evaluation used a deterministic cached environment. That made it useful as a smoke test, but not strong enough as a publication-grade control result.

### Question / confusion

The concern was:

- Does INT8 really preserve control behavior in a meaningful way?
- Or are we just replaying the same frozen scenario repeatedly?

### Method

We strengthened the evaluation by creating a **replay-based control agreement test** on held-out trajectories.

Replay metadata:
- [replay_episodes_6d_strict_metadata.json](/Users/maaary/Downloads/DBS-main/results/strict_eval/replay_episodes_6d_strict_metadata.json)

This test used:

- `50` replay episodes
- `10` steps each
- `500` held-out states total

We then compared FP32 and INT8 action outputs on the exact same held-out trajectories.

### Result

For `96x96`:

- action MAE: `0.0022822`
- action max abs diff: `0.0081592`
- frequency MAE: `0.1223 Hz`
- amplitude MAE: `8.1046`

For comparison, across all three strict models:

| Model | Held-out fidelity MAE | Replay frequency MAE (`Hz`) | Replay amplitude MAE |
| --- | ---: | ---: | ---: |
| `80x80` | `0.002367` | `0.240` | `5.36` |
| `96x96` | `0.002282` | `0.122` | `8.10` |
| `128x128` | `0.002520` | `0.297` | `4.58` |

### Plain-language interpretation

This says:

- INT8 and FP32 chose very similar control actions on many unseen replayed states
- the differences were small in both frequency and amplitude terms
- this is much stronger than the earlier deterministic smoke test

### Limitation / reflection

This is still **replay-based**, not fresh stochastic closed-loop simulation.

So the strongest honest wording is:

> We now have strong held-out action agreement evidence, but not yet a full fresh closed-loop stochastic control equivalence study.

---

## 10. Problem Category 7: Firmware / Runtime Consistency and Benchmark Hygiene

This was a smaller scientific problem, but very important for trust in the numbers.

### Problem

There were multiple runtime paths in the repo, and earlier firmware states were not always aligned with the intended benchmark path.

Examples of confusion included:

- Arduino path vs native ESP-IDF path
- 4D legacy models vs 6D deployment models
- serial output and benchmark runs not always reflecting the host-requested settings

### Question / confusion

The concerns were:

- Are we really comparing the same model family across platforms?
- Is the firmware actually running the run count we think it is?
- Are serial issues making the benchmark look flaky?

### Method

We cleaned this up by:

1. Unifying the main performance story around **6D-only** models.
2. Fixing the native benchmark so requested run counts actually reached the firmware.
3. Making the Arduino sketch's serial output much more compact and benchmark-friendly.
4. Verifying runtime-reported metadata directly from boot/benchmark output.
5. Adding explicit `data_cache_kb` reporting in the native ESP-IDF benchmark output.

### Result

This gave us a much cleaner evidence chain.

For example, the current native `400x300` cache verification line is:

```text
BENCH_RESULT framework=espidf runtime=esp-tflite-micro runs=200 warmup_runs=20 model_io=INT8 data_cache_kb=64 input_dim=6 output_dim=2 quant_avg_us=6.40 invoke_avg_us=10502.52 dequant_avg_us=1.21 total_avg_us=10510.13 min_invoke_us=10501 max_invoke_us=10510
```

This makes it much easier to trust that the benchmark is really running the intended runtime path and cache configuration.

### Limitation / reflection

Arduino is still useful for manual experiments and power measurement, but the **native ESP-IDF path** remains the more paper-faithful deployment benchmark.

---

## 11. What We Learned Overall

### The biggest scientific lessons

1. **INT8 does not automatically become faster on-device.**
   It only becomes faster once the model size and runtime path make INT8 worthwhile.

2. **For this project, `96x96` was the first tested 6D model where INT8 clearly beat FP32 on ESP32-S3.**

3. **The `96x96` speedup is real, but it depends strongly on deployment configuration.**
   In particular, the `32 KB` cache setting matters.

4. **The larger `400x300` model still benefits from INT8, but less strongly.**
   The current evidence suggests that it is more memory-limited.

5. **Fidelity and control-agreement evidence are now much stronger than before.**
   The strict held-out split fixed a major validity problem.

---

## 12. What Is Strong Enough to Show Today

These points are strong enough to present confidently:

- The main deployment candidate is `6D 96x96 INT8`.
- INT8 beat FP32 by `3.87x` on native ESP32-S3 at `32 KB` cache.
- INT8 reduced TFLite model size by `69.65%`.
- INT8 stayed very close to FP32 on a strict held-out fidelity test.
- INT8 also stayed very close to FP32 on replayed held-out trajectories.
- The large `400x300` model also showed an INT8 speedup on ESP32, but the speedup dropped to `1.88x`.
- Cache configuration matters a lot for interpretation.

---

## 13. What Is Still Missing or Should Be Framed Carefully

### Still missing

- A **fully matched** `400x300 FP32 vs INT8 @ 64 KB` cache pair
- A fresh **closed-loop stochastic control** evaluation, not just replay agreement
- If strict replication of the prior paper is required, a careful methods comparison against that paper's exact runtime and memory configuration

### Must be framed carefully

- `96x96` is the best deployment result in **this** setup, not a universal best model
- Desktop TFLite is not a substitute for on-board deployment measurements
- Cache dependence should be explicitly written in the methods and discussion sections

---

## 14. Suggested Questions to Ask the PI Today

1. Is the paper's main goal to optimize **the deployable controller on this ESP32-S3 setup**, or to **replicate the prior `400x300` architecture as closely as possible**?
2. Should the paper's primary result center on `96x96`, with `400x300` as a scaling / failure-mode study?
3. Does the PI want us to add a **fresh closed-loop stochastic test** before submission, or is the current replay-based evidence sufficient for the current milestone?
4. For the larger `400x300` model, does the PI want:
   - a full `64 KB` matched cache experiment,
   - model compression experiments,
   - or strict comparison against the reference paper first?

---

## 15. One-Paragraph Version for a PI Meeting

We found that on the native ESP32-S3 deployment path, `96x96` was the first tested 6D model where INT8 clearly became faster than FP32. Under the default `32 KB` data-cache configuration, INT8 reduced on-device invoke latency from `1685.776 us` to `435.796 us` and reduced model size from `42440` to `12880` bytes. We then repaired the 6D evaluation pipeline, created a strict calibration/held-out split with zero overlap, and confirmed that INT8 stayed very close to FP32 on unseen states and replayed held-out trajectories. We also tested a much larger `400x300` model inspired by the prior paper. INT8 still won on ESP32, but only by `1.88x`, suggesting the model became more memory-limited. The main result is therefore strong but deployment-specific: `96x96 INT8` is the cleanest current candidate for this ESP32-S3 setup, while `400x300` is better framed as a larger-capacity model that exposes memory-system limits.
