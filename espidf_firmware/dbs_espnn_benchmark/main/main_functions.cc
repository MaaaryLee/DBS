#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dbs_benchmark_config.h"
#include "dbs_model_data.h"
#include "main_functions.h"

namespace {

#if CONFIG_ESP32S3_DATA_CACHE_64KB
constexpr int kDataCacheSizeKB = 64;
#elif CONFIG_ESP32S3_DATA_CACHE_32KB
constexpr int kDataCacheSizeKB = 32;
#elif CONFIG_ESP32S3_DATA_CACHE_16KB
constexpr int kDataCacheSizeKB = 16;
#else
constexpr int kDataCacheSizeKB = 0;
#endif

constexpr size_t kTensorArenaSize = static_cast<size_t>(DBS_TENSOR_ARENA_SIZE_BYTES);
constexpr int kMaxObservationDim = 8;
constexpr unsigned long kWarmupRuns = DBS_BENCHMARK_WARMUP_RUNS;
constexpr unsigned long kBenchRuns = DBS_BENCHMARK_RUNS;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

enum ModelIOType {
  kModelIOFloat32 = 0,
  kModelIOInt8 = 1,
  kModelIOUnsupported = 2,
};

ModelIOType model_io_type = kModelIOUnsupported;
bool initialized = false;

int input_feature_count = 0;
int output_feature_count = 0;

float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;

float current_observation[kMaxObservationDim] = {0.0f};
float last_frequency = 0.0f;
float last_amplitude = 0.0f;

uint32_t quant_time_us = 0;
uint32_t invoke_time_us = 0;
uint32_t dequant_time_us = 0;
uint32_t total_time_us = 0;

unsigned long total_inferences = 0;
uint64_t total_quant_time_us = 0;
uint64_t total_invoke_time_us = 0;
uint64_t total_dequant_time_us = 0;
uint64_t total_total_time_us = 0;

uint32_t min_invoke_time_us = UINT32_MAX;
uint32_t max_invoke_time_us = 0;

const char* ModelIOTypeName() {
  switch (model_io_type) {
    case kModelIOFloat32:
      return "FP32";
    case kModelIOInt8:
      return "INT8";
    default:
      return "UNSUPPORTED";
  }
}

bool RegisterModelOps(tflite::MicroMutableOpResolver<2>* resolver) {
  if (resolver->AddFullyConnected() != kTfLiteOk) {
    printf("Failed to register FULLY_CONNECTED\n");
    return false;
  }
  if (resolver->AddTanh() != kTfLiteOk) {
    printf("Failed to register TANH\n");
    return false;
  }
  return true;
}

void ResetTimingStats() {
  total_inferences = 0;
  total_quant_time_us = 0;
  total_invoke_time_us = 0;
  total_dequant_time_us = 0;
  total_total_time_us = 0;
  min_invoke_time_us = UINT32_MAX;
  max_invoke_time_us = 0;
}

void SetDefaultObservation() {
  static const float defaults4[] = {0.5f, 0.3f, 0.7f, 0.2f};
  static const float defaults6[] = {0.5f, 0.3f, 0.7f, 0.2f, 0.1f, 0.4f};

  for (int i = 0; i < kMaxObservationDim; ++i) {
    current_observation[i] = 0.0f;
  }

  const float* source = (input_feature_count >= 6) ? defaults6 : defaults4;
  const int source_count = (input_feature_count >= 6) ? 6 : 4;
  for (int i = 0; i < input_feature_count && i < source_count; ++i) {
    current_observation[i] = source[i];
  }
}

void PrintObservation() {
  printf("Current observation: [");
  for (int i = 0; i < input_feature_count; ++i) {
    printf("%.6f", static_cast<double>(current_observation[i]));
    if (i + 1 < input_feature_count) {
      printf(", ");
    }
  }
  printf("]\n");
}

void PrintModelInfo() {
  printf("=== ESP-IDF DBS Benchmark ===\n");
  printf("framework=espidf runtime=esp-tflite-micro target=esp32s3\n");
  printf("data_cache_kb=%d\n", kDataCacheSizeKB);
  printf("model_len=%u\n", dbs_model_len);
  printf("tensor_arena_bytes=%u\n", static_cast<unsigned int>(kTensorArenaSize));
  printf("input_dim=%d\n", input_feature_count);
  printf("output_dim=%d\n", output_feature_count);
  printf("input_type=%d\n", static_cast<int>(input_tensor->type));
  printf("output_type=%d\n", static_cast<int>(output_tensor->type));
  printf("model_io=%s\n", ModelIOTypeName());
  if (model_io_type == kModelIOInt8) {
    printf("input_scale=%.10f input_zero_point=%d\n",
           static_cast<double>(input_scale), input_zero_point);
    printf("output_scale=%.10f output_zero_point=%d\n",
           static_cast<double>(output_scale), output_zero_point);
  }
  printf("free_heap=%u largest_block=%u\n",
         static_cast<unsigned int>(esp_get_free_heap_size()),
         static_cast<unsigned int>(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT)));
  printf("warmup_runs_configured=%lu bench_runs_configured=%lu\n", kWarmupRuns, kBenchRuns);
  PrintObservation();
  fflush(stdout);
}

bool WriteObservationToInputTensor(const float* observation) {
  const int64_t quant_start_us = esp_timer_get_time();

  if (model_io_type == kModelIOFloat32) {
    for (int i = 0; i < input_feature_count; ++i) {
      input_tensor->data.f[i] = observation[i];
    }
  } else if (model_io_type == kModelIOInt8) {
    for (int i = 0; i < input_feature_count; ++i) {
      int32_t q = static_cast<int32_t>(lrintf(observation[i] / input_scale)) + input_zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input_tensor->data.int8[i] = static_cast<int8_t>(q);
    }
  } else {
    printf("Unsupported model IO mode\n");
    return false;
  }

  quant_time_us = static_cast<uint32_t>(esp_timer_get_time() - quant_start_us);
  return true;
}

bool ReadOutputFromTensor(float* frequency, float* amplitude) {
  const int64_t dequant_start_us = esp_timer_get_time();

  if (model_io_type == kModelIOFloat32) {
    *frequency = output_tensor->data.f[0];
    *amplitude = output_tensor->data.f[1];
  } else if (model_io_type == kModelIOInt8) {
    *frequency = (static_cast<int32_t>(output_tensor->data.int8[0]) - output_zero_point) * output_scale;
    *amplitude = (static_cast<int32_t>(output_tensor->data.int8[1]) - output_zero_point) * output_scale;
  } else {
    printf("Unsupported model IO mode\n");
    return false;
  }

  dequant_time_us = static_cast<uint32_t>(esp_timer_get_time() - dequant_start_us);
  return true;
}

bool RunInference(const float* observation, bool verbose) {
  if (!WriteObservationToInputTensor(observation)) {
    return false;
  }

  const int64_t invoke_start_us = esp_timer_get_time();
  const TfLiteStatus invoke_status = interpreter->Invoke();
  invoke_time_us = static_cast<uint32_t>(esp_timer_get_time() - invoke_start_us);

  if (invoke_status != kTfLiteOk) {
    printf("Invoke failed\n");
    return false;
  }

  if (!ReadOutputFromTensor(&last_frequency, &last_amplitude)) {
    return false;
  }

  total_time_us = quant_time_us + invoke_time_us + dequant_time_us;

  total_inferences += 1;
  total_quant_time_us += quant_time_us;
  total_invoke_time_us += invoke_time_us;
  total_dequant_time_us += dequant_time_us;
  total_total_time_us += total_time_us;
  if (invoke_time_us < min_invoke_time_us) min_invoke_time_us = invoke_time_us;
  if (invoke_time_us > max_invoke_time_us) max_invoke_time_us = invoke_time_us;

  if (verbose) {
    printf("last_output0=%.6f last_output1=%.6f\n",
           static_cast<double>(last_frequency), static_cast<double>(last_amplitude));
    printf("last_quant_us=%" PRIu32 " last_invoke_us=%" PRIu32 " last_dequant_us=%" PRIu32
           " last_total_us=%" PRIu32 "\n",
           quant_time_us, invoke_time_us, dequant_time_us, total_time_us);
  }

  return true;
}

void PrintBenchResult(unsigned long runs) {
  const float avg_quant_us = (total_inferences > 0)
                                 ? static_cast<float>(total_quant_time_us) / static_cast<float>(total_inferences)
                                 : 0.0f;
  const float avg_invoke_us = (total_inferences > 0)
                                  ? static_cast<float>(total_invoke_time_us) / static_cast<float>(total_inferences)
                                  : 0.0f;
  const float avg_dequant_us = (total_inferences > 0)
                                   ? static_cast<float>(total_dequant_time_us) / static_cast<float>(total_inferences)
                                   : 0.0f;
  const float avg_total_us = (total_inferences > 0)
                                 ? static_cast<float>(total_total_time_us) / static_cast<float>(total_inferences)
                                 : 0.0f;

  printf(
      "BENCH_RESULT framework=espidf runtime=esp-tflite-micro runs=%lu warmup_runs=%lu model_io=%s "
      "data_cache_kb=%d input_dim=%d output_dim=%d quant_avg_us=%.2f invoke_avg_us=%.2f dequant_avg_us=%.2f total_avg_us=%.2f "
      "min_invoke_us=%" PRIu32 " max_invoke_us=%" PRIu32 "\n",
      runs,
      kWarmupRuns,
      ModelIOTypeName(),
      kDataCacheSizeKB,
      input_feature_count,
      output_feature_count,
      static_cast<double>(avg_quant_us),
      static_cast<double>(avg_invoke_us),
      static_cast<double>(avg_dequant_us),
      static_cast<double>(avg_total_us),
      min_invoke_time_us == UINT32_MAX ? 0 : min_invoke_time_us,
      max_invoke_time_us);
  printf("BENCH_DONE\n");
  fflush(stdout);
}

void RunBenchmark(unsigned long warmups, unsigned long runs) {
  for (unsigned long i = 0; i < warmups; ++i) {
    if (!RunInference(current_observation, false)) {
      return;
    }
  }

  ResetTimingStats();
  for (unsigned long i = 0; i < runs; ++i) {
    if (!RunInference(current_observation, false)) {
      return;
    }
  }

  PrintBenchResult(runs);
}

}  // namespace

void setup() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  tflite::InitializeTarget();

  model = tflite::GetModel(dbs_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Model schema version %" PRIu32 " not supported. Supported is %d.\n",
           model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<2> resolver;
  if (!RegisterModelOps(&resolver)) {
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors() failed\n");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  input_feature_count = input_tensor->dims->data[input_tensor->dims->size - 1];
  output_feature_count = output_tensor->dims->data[output_tensor->dims->size - 1];

  if (input_feature_count <= 0 || input_feature_count > kMaxObservationDim) {
    printf("Input dimension %d exceeds sketch limit %d\n", input_feature_count, kMaxObservationDim);
    return;
  }

  if (input_tensor->type == kTfLiteFloat32 && output_tensor->type == kTfLiteFloat32) {
    model_io_type = kModelIOFloat32;
  } else if (input_tensor->type == kTfLiteInt8 && output_tensor->type == kTfLiteInt8) {
    model_io_type = kModelIOInt8;
    input_scale = input_tensor->params.scale;
    input_zero_point = input_tensor->params.zero_point;
    output_scale = output_tensor->params.scale;
    output_zero_point = output_tensor->params.zero_point;
  } else {
    model_io_type = kModelIOUnsupported;
    printf("Unsupported input/output tensor types\n");
    return;
  }

  SetDefaultObservation();
  PrintModelInfo();
  initialized = true;
}

void loop() {
  if (!initialized) {
    printf("Benchmark not initialized\n");
    fflush(stdout);
    return;
  }

  RunBenchmark(kWarmupRuns, kBenchRuns);
}
