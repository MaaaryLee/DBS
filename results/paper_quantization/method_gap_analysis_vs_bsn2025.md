# Quantization Study Method and Remaining Gaps vs. the BSN 2025 DBS Paper

Reference paper reviewed here:
- [v2-Deep_Brain_Stimulation_BSN_2025-1.pdf](/Users/maaary/Library/Mobile%20Documents/com~apple~CloudDocs/Desktop/Fall%2025'/COMP%20691/v2-Deep_Brain_Stimulation_BSN_2025-1.pdf)

This note answers two questions:

1. **What research method should this quantization paper follow?**
2. **What is still missing before we can claim that quantization improved the controller in a publishable way?**

The language here is intentionally plain. It is written so a new student or non-specialist PI can follow the logic.

---

## 1. What the BSN 2025 Paper Actually Did

The reference paper is **not** a quantization paper.

Its core contribution is:

- train a **TD3 DBS controller**
- use **measurable biomarkers** instead of simulation-only error index features
- compare the RL controller against a standard **open-loop DBS (o-DBS)** baseline

### Biomarkers / state used in the paper

The paper defines a **6D state** using:

- standard deviation of `SGi`
- Hjorth activity of `SGi`
- Hjorth mobility of `SGi`
- Hjorth complexity of `SGi`
- beta-band PSD of `VGi`
- sample entropy of STN activity

This is very close to the 6D feature story we have been using in the repo.

### Action space in the paper

The controller adjusts:

- frequency in `[0, 185] Hz`
- amplitude in `[0, 5000] µA/cm²`

These are normalized to `[-1, 1]` for the RL policy.

### Reward in the paper

The paper optimizes two things:

- reduce PD-related biomarker severity, mainly `SGi PSD`
- reduce power usage

The paper explicitly prioritizes biomarker suppression more than power, but still rewards lower power.

### Reported end results in the paper

The paper compares its TD3 controller to o-DBS and reports:

| Metric | o-DBS | TD3-DBS | Improvement |
| --- | ---: | ---: | ---: |
| `SGi PSD` | `1360` | `1260` | `7.35%` lower |
| `VGi PSD` | `361000` | `336000` | `6.93%` lower |
| Power usage | `494` | `341` | `31%` lower |

Important interpretation:

> The BSN paper's novelty is better **control behavior** using measurable biomarkers.
> It does **not** study FP32 vs INT8, model size, TFLite latency, or ESP32 deployment.

That means your paper should be framed as an **extension toward low-power deployment**, not as a repetition of exactly the same contribution.

---

## 2. What Your Paper Can Contribute

Your paper can naturally extend the BSN work in this direction:

> If we already have an RL DBS controller that works in simulation with measurable biomarkers, can we make it more deployable by quantizing it, while preserving its behavior?

That leads to a clean paper goal:

> Study whether post-training quantization reduces deployment cost
> while keeping the DBS controller behavior close to the FP32 baseline.

This is a stronger and more precise goal than simply saying:

> \"INT8 is faster.\"

Instead, the claim should be:

> Quantization improves **deployment efficiency** with minimal degradation in controller behavior.

---

## 3. Recommended Research Method for Your Paper

This is the method I would recommend writing into the paper.

### Step 1. Define the scientific question

Use a question like this:

> For RL-based intelligent DBS controllers using measurable 6D biomarkers, can INT8 quantization reduce model size and inference cost while preserving control behavior relative to the FP32 controller?

This is important because it makes the paper about:

- deployment
- low-power / low-cost inference
- behavior preservation

not just raw speed.

### Step 2. Fix the baseline controller

You need one clear FP32 reference model.

Right now, the cleanest choice is:

- **Primary deployment candidate:** 6D `96x96`

Reason:

- it is the first tested 6D model where INT8 beat FP32 on native ESP32-S3
- it already has the strongest supporting evidence bundle

You can keep a second, larger model as a scaling study:

- **Secondary scaling model:** 6D `400x300`

But that should be secondary unless your PI specifically wants the whole paper centered on the larger architecture.

Important note:

The reviewed BSN paper PDF does **not** clearly specify a `400x300` actor architecture in the text we extracted. So if you use `400x300`, describe it as:

- a **paper-inspired larger controller**
- or a **larger comparison architecture**

Do **not** imply that the reviewed PDF explicitly defines `400x300` unless you can cite that exact line from another source.

### Step 3. Quantize the controller

For each chosen architecture:

1. train or load the FP32 TD3 controller
2. export FP32 TFLite
3. export INT8 TFLite using representative calibration states
4. keep the same input features, output normalization, and control ranges

This makes FP32 vs INT8 a fair numerical comparison of the same controller family.

### Step 4. Evaluate along four dimensions

Your paper should report **four kinds of evidence**.

#### 4A. Model compactness

Report:

- file size in bytes
- percent size reduction
- parameter count
- MACs per inference

Why:

- quantization is partly about memory footprint, not just latency

#### 4B. Inference efficiency

Report separately:

- desktop TFLite latency
- native ESP32 latency

For ESP32, report:

- `invoke_avg_us`
- `total_avg_us`
- runtime path
- cache setting
- run count / repeats

Why:

- desktop TFLite is a software-side reference
- ESP32 is the deployment-relevant result

#### 4C. Fidelity to FP32

Report:

- mean absolute error (MAE)
- max absolute difference
- per-output Pearson correlation
- input saturation rate

Run this on a **strict held-out state set disjoint from calibration**.

Why this matters:

- calibration data is what the quantizer used to set numeric ranges
- if you test on the same data, you may overestimate how well INT8 behaves
- a held-out test is more like an exam instead of practicing on the answer sheet

#### 4D. Control behavior preservation

This is the most important part if you want to connect back to the BSN paper.

The BSN paper's success is about **control quality**, not just speed.

So your quantization paper should eventually show that the quantized controller still preserves the **control outcome**, not only the raw action values.

The strongest version is:

- run fresh closed-loop evaluation in the simulator
- compare FP32 vs INT8 on the same episode settings
- report biomarker outcomes like:
  - `SGi PSD`
  - `VGi PSD`
  - power / energy usage
  - maybe reward, if used carefully

This is the missing bridge between:

- \"INT8 matches FP32 numerically\"
- and
- \"INT8 preserves the actual DBS control benefit\"

### Step 5. State the runtime conditions explicitly

This part is especially important in your project.

You must write:

- board: Arduino Nano ESP32 / ESP32-S3
- runtime: `ESP-IDF + esp-tflite-micro + ESP-NN`
- cache setting: `32 KB` or `64 KB`
- model input dimension: 6D
- number of benchmark runs and repeats

Why:

- your results are sensitive to deployment configuration
- if this is omitted, readers may think the speedup is universal

### Step 6. Separate main claim from supporting claim

Main claim:

> INT8 improves deployment efficiency for the selected DBS controller under the tested ESP32-S3 runtime configuration while maintaining close agreement with FP32.

Supporting claim:

> The size of the speedup depends on model size and memory configuration.

This structure is much safer than trying to claim:

> INT8 is simply better in all cases.

---

## 4. What You Have Already Completed

You already have several strong pieces of the method done.

### A. Strong deployment evidence for the 6D `96x96` candidate

For `96x96`:

- desktop TFLite:
  - FP32 `0.0016416 ms`
  - INT8 `0.0013336 ms`
  - INT8 `1.23x` faster
- native ESP32-S3, `32 KB` cache:
  - FP32 `1685.776 us`
  - INT8 `435.796 us`
  - INT8 `3.87x` faster
- model size:
  - FP32 `42440` bytes
  - INT8 `12880` bytes
  - `69.65%` reduction

### B. Strong strict held-out fidelity evidence

For `96x96` on a disjoint held-out set:

- overlap between calibration and held-out source rows: `0`
- fidelity MAE: `0.0022822`
- max abs diff: `0.0081592`
- Pearson r: `0.999923`, `0.999831`

This is already a strong argument that INT8 stays close to FP32 on unseen inputs.

### C. Better-than-smoke-test control agreement evidence

For `96x96`, replay-based held-out agreement showed:

- action MAE: `0.0022822`
- frequency MAE: `0.1223 Hz`
- amplitude MAE: `8.1046`

This is much stronger than the earlier deterministic smoke test.

### D. Scaling-study evidence

You also showed that:

- `80x80` was too small for INT8 to win on native ESP32-S3
- `96x96` was the first tested winner
- `128x128` also won
- `400x300` still won in INT8, but less strongly

This is useful because it tells a realistic deployment story: larger models do not automatically mean better INT8 speedup.

---

## 5. What Is Still Missing Before You Can Claim a Quantization Improvement

This is the most important section.

### Missing item 1. A true closed-loop control comparison using the BSN-style outcome metrics

This is the biggest gap.

Right now, you have:

- strict held-out fidelity
- replay-based control agreement

Those are very valuable, but they still do **not** directly show:

- whether INT8 preserves `SGi PSD`
- whether INT8 preserves `VGi PSD`
- whether INT8 preserves the DBS power/energy behavior

That matters because the BSN paper's core claim is about **improved control outcomes relative to o-DBS**.

So if your paper wants to claim that quantization preserves or improves the controller, you still need:

> a closed-loop FP32 vs INT8 comparison on the same DBS evaluation metrics used by the DBS paper

At minimum, I would want:

- `SGi PSD`
- `VGi PSD`
- power / energy use
- same episode design
- mean ± std across multiple episodes / seeds

### Missing item 2. Real power or energy measurements on the deployment target

Your PI explicitly described the project as exploring **low-power variants**.

Right now, you already have:

- size reduction
- latency reduction

But to claim **low-power improvement** in a stronger way, you still need one of:

- on-board current / power measurements
- energy per inference
- or at least a well-labeled board-side proxy measurement

Without this, the paper can safely claim:

- improved size
- improved latency
- improved deployment efficiency

But it is weaker if it claims:

- improved power

### Missing item 3. Clear separation between the main candidate and the scaling study

Right now, the cleanest evidence bundle is for `96x96`.

The `400x300` model is useful, but it is more complicated:

- INT8 speedup is smaller
- memory effects are stronger
- it is not currently the neatest deployment candidate

So the paper should decide:

- either `96x96` is the main result and `400x300` is a scaling / stress test
- or the paper commits to a full paper-inspired `400x300` replication study

If you try to make both the main story, the message will get muddy.

### Missing item 4. A precise statement of what kind of \"improvement\" you are claiming

This is subtle but very important.

You can already support:

> quantization improves deployment efficiency while preserving controller behavior closely

You cannot yet fully support:

> quantization improves therapeutic DBS performance

Why not?

Because the therapeutic-style control metrics from the BSN paper have not yet been rerun in a strong FP32-vs-INT8 closed-loop comparison.

### Missing item 5. If you want to tie directly to the BSN paper, you need a comparison paragraph that is careful

The safest comparison is:

- BSN paper:
  RL controller vs o-DBS using measurable biomarkers
- your paper:
  FP32 controller vs INT8 controller for deployment efficiency and behavior preservation

That is a valid extension.

It is weaker if you imply:

- you reproduced all of their control claims exactly

unless you actually rerun the same outcome metrics with the same evaluation style.

---

## 6. What You Can Safely Claim Right Now

If you wrote the paper today, these claims are already defensible.

### Safe claim 1

> For the 6D `96x96` controller, INT8 quantization substantially reduced model size and on-device inference latency relative to FP32 on the tested ESP32-S3 deployment stack.

### Safe claim 2

> On a strict held-out state split, the INT8 policy remained very close to FP32 in output behavior.

### Safe claim 3

> On replayed held-out trajectories, the INT8 and FP32 policies produced very similar control actions.

### Safe claim 4

> The deployment benefit of INT8 depended on model size and runtime configuration; the `96x96` model was the first tested 6D size where INT8 clearly outperformed FP32 on-device.

---

## 7. What You Should Not Claim Yet

Avoid these claims unless you add more experiments.

### Not safe yet

- \"Quantization improved DBS therapeutic performance.\"
- \"INT8 preserved all original RL control benefits from the BSN paper.\"
- \"INT8 is universally faster on ESP32-S3.\"
- \"The larger `400x300` model is the best deployment model.\"
- \"We demonstrated low-power improvement\" without real board-side power/energy measurements.

---

## 8. Recommended Paper Framing

This is the framing I would recommend.

### Suggested paper story

1. Start from the BSN-style measurable-biomarker DBS controller idea.
2. Reproduce the controller pipeline in the Python-MATLAB environment.
3. Ask a new question:
   can this controller be made more deployable through quantization?
4. Compare FP32 vs INT8 in:
   - model size
   - desktop TFLite latency
   - native ESP32 latency
   - strict held-out fidelity
   - replay-based control agreement
5. Use `96x96` as the main deployment candidate.
6. Use `400x300` as a larger-model stress test that reveals memory bottlenecks.

### Suggested main paper claim

> We extend prior RL-based intelligent DBS work toward deployment by showing that post-training INT8 quantization can substantially reduce model size and on-device inference latency for a measurable-biomarker TD3 controller, while preserving close agreement with the FP32 policy on held-out states and replayed trajectories.

This is much safer than claiming a new clinical-control improvement.

---

## 9. Best Next Experiments

If the goal is to make the paper clearly publishable, I would prioritize these next:

1. **Closed-loop FP32 vs INT8 evaluation using BSN-style outcome metrics**
   - `SGi PSD`
   - `VGi PSD`
   - power / energy use

2. **On-board power or energy measurement**
   - same model
   - same runtime
   - FP32 vs INT8

3. **Choose the main story**
   - `96x96` as main candidate
   - `400x300` as scaling study

4. **Methods paragraph that explicitly states the deployment stack**
   - board
   - runtime
   - cache setting
   - repeats
   - run count

---

## 10. Bottom Line

Your current work is already strong enough to support a **deployment-oriented quantization paper**.

But if you want to claim a true **improvement of quantization** in the context of the BSN DBS study, the strongest missing piece is:

> showing that the quantized controller preserves the BSN-style closed-loop biomarker outcomes, not just its raw action outputs.

So the short answer is:

- **Yes**, you already have a strong deployment story.
- **No**, you do not yet have the full evidence needed to say that quantization preserved the original DBS therapeutic-control benefit as strongly as possible.

That final bridge needs:

- closed-loop biomarker evaluation
- and ideally board-side power / energy measurement
