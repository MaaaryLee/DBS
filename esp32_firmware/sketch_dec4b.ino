#include "ardutflite_compat.h"
#include "model_int_8.h"
#include <Arduino.h>
// extern "C" {
// #include "esp32s3/rom/cache.h"
// }

constexpr int kDbsTensorArenaSize = 32 * 1024;
constexpr int kExpectedInputSize = 4;
alignas(16) uint8_t dbs_tensor_arena[kDbsTensorArenaSize];


void setup() {
  // 1. Initialize Serial first for debugging
  Serial.begin(115200);

  // 2. Wait for USB Serial to connect (Native USB handshake) 
  // 原生USB：给主机一点枚举的时间 （3000ms） （可以改成2000ms）
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 3000)) { delay(10); }

  Serial.println("boot: start");
  Serial.print("model_len = ");
  Serial.println(model_len);
  Serial.print("tensor_arena_bytes = ");
  Serial.println(kDbsTensorArenaSize);

  bool ok = modelInit(model, dbs_tensor_arena, kDbsTensorArenaSize);
  Serial.print("modelInit returned: ");
  Serial.println(ok ? "true" : "false");

  if (!ok) {
    Serial.print("modelInit failed: ");
    Serial.println(modelGetLastError());
    // 关键：加 delay，避免 watchdog reset 导致反复重启
    while (true) { delay(1000); }
  }

  Serial.println("modelInit ok");
  Serial.print("input_len = ");
  Serial.println(modelGetInputLength());
  Serial.print("output_len = ");
  Serial.println(modelGetOutputLength());
  Serial.print("input_type = ");
  Serial.println(modelInputIsInt8() ? "INT8" : "FP32/OTHER");
  Serial.print("output_type = ");
  Serial.println(modelOutputIsInt8() ? "INT8" : "FP32/OTHER");

  if (modelGetInputLength() != kExpectedInputSize) {
    Serial.print("This Hjorth benchmark computes ");
    Serial.print(kExpectedInputSize);
    Serial.print(" features, but the flashed model expects ");
    Serial.print(modelGetInputLength());
    Serial.println(".");
    Serial.println("Use the 4D model for this sketch, or switch to dbs_inference.ino for a dimension-flexible path.");
    while (true) { delay(1000); }
  }

  if (modelGetOutputLength() < 2) {
    Serial.println("Model output tensor is smaller than expected.");
    while (true) { delay(1000); }
  }

  randomSeed(micros());
}

void loop() {


  static bool done = false;
  if (done) { delay(1000); return; }

  // 1x4 model input: (sd, A, M, C)
  constexpr int INPUT_SIZE = kExpectedInputSize;
  constexpr int NSAMPLES   = 250;

  constexpr int WARMUP = 20;
  constexpr int N      = 500;

  auto clamp01 = [](float v) -> float {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
  };

  // Welford accumulator (biased variance = M2 / n)
  struct Welford {
    int n = 0;
    float mean = 0.0f;
    float M2 = 0.0f;
  };

  auto welford_update = [](Welford &s, float x) {
    s.n += 1;
    float delta = x - s.mean;
    s.mean += delta / (float)s.n;
    float delta2 = x - s.mean;
    s.M2 += delta * delta2;
  };

  auto welford_var_biased = [](const Welford &s) -> float {
    if (s.n <= 0) return 0.0f;
    return s.M2 / (float)s.n; // biased variance
  };

  // Normalization constants (use your training constants)
  constexpr float sd_min = 25.8499f, sd_max = 32.3666f;
  constexpr float A_min  = 672.2282f, A_max  = 1047.6699f;
  constexpr float M_min  = 0.0920f,  M_max  = 0.0969f;
  constexpr float C_min  = 0.7911f,  C_max  = 0.8618f;

  constexpr float inv_sd = 1.0f / (sd_max - sd_min);
  constexpr float inv_A  = 1.0f / (A_max  - A_min);
  constexpr float inv_M  = 1.0f / (M_max  - M_min);
  constexpr float inv_C  = 1.0f / (C_max  - C_min);

  constexpr float eps = 1e-6f;

  uint64_t preprocess_us = 0;
  uint64_t setinput_us   = 0;
  uint64_t infer_us      = 0;
  uint64_t total_us      = 0;

  // -----------------------
  // WARMUP (not timed)
  // -----------------------
  for (int it = 0; it < WARMUP; it++) {
    float d0_0 = 0.0f, d0_1 = 0.0f, d0_2 = 0.0f;
    Welford s0, s1, s2;

    for (int i = 0; i < NSAMPLES; i++) {
      d0_0 = d0_1;
      d0_1 = d0_2;
      d0_2 = (float)random(-80000, 80000) / 1000.0f; // replace with sensor later

      float d1 = d0_1 - d0_0;
      float d2 = (d0_2 - d0_1) - (d0_1 - d0_0);

      welford_update(s0, d0_2);
      welford_update(s1, d1);
      welford_update(s2, d2);
    }

    float var0 = welford_var_biased(s0);
    float var1 = welford_var_biased(s1);
    float var2 = welford_var_biased(s2);

    // Hjorth parameters (standard)
    float A_val  = var0;
    float sd_val = sqrtf((A_val > 0.0f) ? A_val : 0.0f);
    float mobility = sqrtf(var1 / (A_val + eps));
    float complexity = sqrtf(var2 / (var1 + eps)) / (mobility + eps);

    float x[INPUT_SIZE];
    x[0] = clamp01((sd_val     - sd_min) * inv_sd);
    x[1] = clamp01((A_val      - A_min)  * inv_A);
    x[2] = clamp01((mobility   - M_min)  * inv_M);
    x[3] = clamp01((complexity - C_min)  * inv_C);

    for (int i = 0; i < INPUT_SIZE; i++) {
      if (!modelSetInput(x[i], i)) {
        Serial.print("warmup modelSetInput failed: ");
        Serial.println(modelGetLastError());
        while (true) { delay(1000); }
      }
    }
    if (!modelRunInference()) {
      Serial.print("warmup inference failed: ");
      Serial.println(modelGetLastError());
      while (true) { delay(1000); }
    }
  }

  // -----------------------
  // BENCHMARK
  // -----------------------
  for (int it = 0; it < N; it++) {
    uint32_t t0 = micros();

    // --- Feature extraction over a 250-sample window ---
    float d0_0 = 0.0f, d0_1 = 0.0f, d0_2 = 0.0f;
    Welford s0, s1, s2;

    for (int i = 0; i < NSAMPLES; i++) {
      d0_0 = d0_1;
      d0_1 = d0_2;
      d0_2 = (float)random(-80000, 80000) / 1000.0f;

      float d1 = d0_1 - d0_0;
      float d2 = (d0_2 - d0_1) - (d0_1 - d0_0);

      welford_update(s0, d0_2);
      welford_update(s1, d1);
      welford_update(s2, d2);
    }

    float var0 = welford_var_biased(s0);
    float var1 = welford_var_biased(s1);
    float var2 = welford_var_biased(s2);

    // Hjorth parameters (standard)
    float A_val  = var0; // Activity
    float sd_val = sqrtf((A_val > 0.0f) ? A_val : 0.0f);
    float mobility = sqrtf(var1 / (A_val + eps));
    float complexity = sqrtf(var2 / (var1 + eps)) / (mobility + eps);

    // Normalize -> model input
    float x[INPUT_SIZE];
    x[0] = clamp01((sd_val     - sd_min) * inv_sd);
    x[1] = clamp01((A_val      - A_min)  * inv_A);
    x[2] = clamp01((mobility   - M_min)  * inv_M);
    x[3] = clamp01((complexity - C_min)  * inv_C);

    uint32_t t_pre_done = micros();

    // Set inputs
    for (int i = 0; i < INPUT_SIZE; i++) {
      if (!modelSetInput(x[i], i)) {
        Serial.print("modelSetInput failed: ");
        Serial.println(modelGetLastError());
        while (true) { delay(1000); }
      }
    }

    uint32_t t_set_done = micros();

    // Inference
    if (!modelRunInference()) {
      Serial.print("modelRunInference failed: ");
      Serial.println(modelGetLastError());
      while (true) { delay(1000); }
    }

    uint32_t t_inf_done = micros();

    // Unsigned deltas handle micros() wrap
    preprocess_us += (uint32_t)(t_pre_done - t0);
    setinput_us   += (uint32_t)(t_set_done - t_pre_done);
    infer_us      += (uint32_t)(t_inf_done - t_set_done);
    total_us      += (uint32_t)(t_inf_done - t0);

    delay(1); // helps native USB stability
  }

  // -----------------------
  // PRINT SUMMARY ONCE
  // -----------------------
  Serial.println("--- Benchmark Complete ---");
  Serial.print("N = "); Serial.println(N);

  Serial.print("avg_preprocess_us = ");
  Serial.println((double)preprocess_us / (double)N, 3);

  Serial.print("avg_setInput_us   = ");
  Serial.println((double)setinput_us / (double)N, 3);

  Serial.print("avg_infer_us      = ");
  Serial.println((double)infer_us / (double)N, 3);

  Serial.print("avg_total_us      = ");
  Serial.println((double)total_us / (double)N, 3);

  Serial.print("out0 = "); Serial.println(modelGetOutput(0), 6);
  Serial.print("out1 = "); Serial.println(modelGetOutput(1), 6);

  Serial.print("BENCH_RESULT runs=");
  Serial.print(N);
  Serial.print(" input_dim=");
  Serial.print(modelGetInputLength());
  Serial.print(" output_dim=");
  Serial.print(modelGetOutputLength());
  Serial.print(" model_io=");
  Serial.print(modelInputIsInt8() && modelOutputIsInt8() ? "INT8" : "FP32");
  Serial.print(" avg_preprocess_us=");
  Serial.print((double)preprocess_us / (double)N, 3);
  Serial.print(" avg_setinput_us=");
  Serial.print((double)setinput_us / (double)N, 3);
  Serial.print(" avg_infer_us=");
  Serial.print((double)infer_us / (double)N, 3);
  Serial.print(" avg_total_us=");
  Serial.println((double)total_us / (double)N, 3);
  Serial.println("BENCH_DONE");

  done = true;
}
