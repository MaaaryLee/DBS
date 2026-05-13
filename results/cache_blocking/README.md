# Cache-Blocking Baseline

This folder stores the first standalone C++ dense-layer experiments for cache blocking / tiling.

Purpose:

- verify that blocked math matches the conventional dense layer numerically
- measure whether a blocked dense kernel helps on the computer before touching the ESP32 firmware path
- separate **kernel-level** effects from **TFLite / interpreter** effects

Important scope note:

- this is **not** an end-to-end ESP32 result
- this is a **standalone C++ dense-kernel baseline** on the computer
- the goal is to decide whether cache-aware dense execution is promising enough to port into firmware

## Files

- [cache_blocking_96x96_rt8_ct32_20260420_112153.json](/Users/maaary/Downloads/DBS-main/results/cache_blocking/cache_blocking_96x96_rt8_ct32_20260420_112153.json)
  6D `96x96` model-shape kernel benchmark with `row_tile=8`, `col_tile=32`
- [cache_blocking_400x300_rt8_ct32_20260420_112153.json](/Users/maaary/Downloads/DBS-main/results/cache_blocking/cache_blocking_400x300_rt8_ct32_20260420_112153.json)
  6D `400x300` model-shape kernel benchmark with warm-cache timing
- [cache_blocking_400x300_rt8_ct32_20260420_112205.json](/Users/maaary/Downloads/DBS-main/results/cache_blocking/cache_blocking_400x300_rt8_ct32_20260420_112205.json)
  same `400x300` benchmark, but with a `262144`-byte flush buffer touched before each timed call to reduce warm-cache bias

## Exact Command

Compile-and-run helper:

```bash
python3 scripts/run_cache_blocking_benchmark.py \
  --preset 400x300 \
  --row-tile 8 \
  --col-tile 32 \
  --warmup 200 \
  --repeats 3000
```

Cold-ish memory variant:

```bash
python3 scripts/run_cache_blocking_benchmark.py \
  --preset 400x300 \
  --row-tile 8 \
  --col-tile 32 \
  --warmup 50 \
  --repeats 1000 \
  --flush-bytes 262144
```

## Initial Findings

### 6D `400x300`, warm-cache run

- Total kernel time:
  naive `76449.139 ns`
- Total kernel time:
  blocked row-major `29183.542 ns`
- Total kernel time:
  blocked packed `29358.639 ns`
- Middle `fc2` layer:
  naive `75095.278 ns`
- Middle `fc2` layer:
  blocked row-major `27522.847 ns`
- Middle `fc2` layer:
  blocked packed `27243.944 ns`
- Numerical agreement:
  max abs diff was `0.0` for both blocked variants against the naive baseline

Interpretation:

- the big middle layer is where blocking helps
- that matches the working hypothesis that the large dense layer is the right place to focus

### 6D `400x300`, flush-buffer run

- Total kernel time:
  naive `75592.795 ns`
- Total kernel time:
  blocked row-major `30615.457 ns`
- Total kernel time:
  blocked packed `30247.969 ns`
- Middle `fc2` layer:
  naive `74091.865 ns`
- Middle `fc2` layer:
  blocked row-major `28748.881 ns`
- Middle `fc2` layer:
  blocked packed `28187.507 ns`

Interpretation:

- the blocked advantage still remained after adding a simple pre-run memory flush
- that makes the result harder to dismiss as only a warm-desktop-cache artifact

### 6D `96x96`, warm-cache run

- Total kernel time:
  naive `4682.283 ns`
- Total kernel time:
  blocked row-major `2852.317 ns`
- Total kernel time:
  blocked packed `2697.000 ns`
- Middle `fc2` layer:
  naive `4365.025 ns`
- Middle `fc2` layer:
  blocked row-major `2357.867 ns`
- Middle `fc2` layer:
  blocked packed `2272.625 ns`
- Numerical agreement:
  max abs diff was `0.0`

Interpretation:

- blocking also helped on the smaller 6D model
- the effect is still driven mainly by the middle fully connected layer

## What This Does And Does Not Mean

What it means:

- a cache-aware dense kernel is worth taking seriously
- the large fully connected layer is a real optimization target
- pre-packed tile layout may help, but in this first pass it was only slightly different from blocked row-major

What it does **not** mean:

- it does not prove the same speedup will appear automatically inside TFLite Micro
- it does not replace the native ESP32 benchmark
- it does not yet tell us the best tile size for ESP32-S3

## Next Step

The next clean step is:

1. sweep tile sizes on the standalone benchmark
2. choose the best candidate for the large middle layer
3. decide whether to port that idea into a custom firmware-side dense kernel or keep it as a supporting systems experiment
