/*
 * ESP32 DBS Inference Benchmark Sketch
 *
 * This sketch is benchmark-oriented:
 * 1. Loads the TFLite model embedded in model.h
 * 2. Detects input/output tensor shapes and dtypes at runtime
 * 3. Accepts observations over Serial so 4D and 6D models can be tested fairly
 * 4. Reports quantize / invoke / dequant timings separately
 *
 * Serial commands (115200 baud):
 *   help
 *   info
 *   defaults
 *   sample <v1> <v2> ... <vN>
 *   run
 *   bench <count>
 */

#include "model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <esp32-hal-psram.h>
#include <esp_heap_caps.h>

namespace {

// Larger 6D models such as 400x300 need substantially more workspace than 96x96.
constexpr int kTensorArenaSize = 256 * 1024;
constexpr int kTensorArenaAlignment = 16;
constexpr int kMaxObservationDim = 8;
constexpr unsigned long kSerialAttachWaitMs = 3000;
constexpr unsigned long kSerialSettleDelayMs = 250;

uint8_t* tensor_arena = nullptr;

}  // namespace

tflite::MicroMutableOpResolver<2> resolver;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

enum ModelIOType {
  kModelIOFloat32 = 0,
  kModelIOInt8 = 1,
  kModelIOUnsupported = 2,
};

ModelIOType model_io_type = kModelIOUnsupported;

int input_feature_count = 0;
int output_feature_count = 0;

float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;

float current_observation[kMaxObservationDim] = {0.0f};

size_t free_heap_before = 0;
size_t free_heap_after = 0;

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

uint32_t min_quant_time_us = UINT32_MAX;
uint32_t max_quant_time_us = 0;
uint32_t min_invoke_time_us = UINT32_MAX;
uint32_t max_invoke_time_us = 0;
uint32_t min_dequant_time_us = UINT32_MAX;
uint32_t max_dequant_time_us = 0;
uint32_t min_total_time_us = UINT32_MAX;
uint32_t max_total_time_us = 0;

const char* modelIOTypeName() {
  switch (model_io_type) {
    case kModelIOFloat32:
      return "FP32";
    case kModelIOInt8:
      return "INT8";
    default:
      return "UNSUPPORTED";
  }
}

bool registerModelOps() {
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("Failed to register FULLY_CONNECTED.");
    return false;
  }
  if (resolver.AddTanh() != kTfLiteOk) {
    Serial.println("Failed to register TANH.");
    return false;
  }
  return true;
}

void resetTimingStats() {
  total_inferences = 0;
  total_quant_time_us = 0;
  total_invoke_time_us = 0;
  total_dequant_time_us = 0;
  total_total_time_us = 0;

  min_quant_time_us = UINT32_MAX;
  max_quant_time_us = 0;
  min_invoke_time_us = UINT32_MAX;
  max_invoke_time_us = 0;
  min_dequant_time_us = UINT32_MAX;
  max_dequant_time_us = 0;
  min_total_time_us = UINT32_MAX;
  max_total_time_us = 0;
}

void printCurrentObservation() {
  Serial.print("OBS ");
  for (int i = 0; i < input_feature_count; ++i) {
    Serial.print(current_observation[i], 6);
    if (i + 1 < input_feature_count) {
      Serial.print(" ");
    }
  }
  Serial.println();
}

void setDefaultObservation() {
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

void printHelp() {
  Serial.println("HELP commands=help,info,defaults,sample,run,bench");
}

void printModelInfo() {
  Serial.printf("INFO model_len=%u input_dim=%d output_dim=%d input_type=%d output_type=%d model_io=%s tensor_arena_bytes=%d\n",
                dbs_model_len,
                input_tensor->dims->data[input_tensor->dims->size - 1],
                output_tensor->dims->data[output_tensor->dims->size - 1],
                static_cast<int>(input_tensor->type),
                static_cast<int>(output_tensor->type),
                modelIOTypeName(),
                kTensorArenaSize);
  if (model_io_type == kModelIOInt8) {
    Serial.printf("INFO_INT8 input_scale=%.10f input_zero_point=%d output_scale=%.10f output_zero_point=%d\n",
                  input_scale,
                  input_zero_point,
                  output_scale,
                  output_zero_point);
  }
  printCurrentObservation();
}

void printMemoryStats(const char* label) {
  const size_t free_heap = esp_get_free_heap_size();
  const size_t min_free_heap = esp_get_minimum_free_heap_size();
  const size_t largest_block = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  const int32_t heap_delta = static_cast<int32_t>(free_heap_before) - static_cast<int32_t>(free_heap_after);

  Serial.printf("MEM label=%s free_heap=%u free_heap_kb=%.2f",
                label,
                static_cast<unsigned int>(free_heap), free_heap / 1024.0f);
  Serial.printf(" min_free_heap=%u min_free_heap_kb=%.2f",
                static_cast<unsigned int>(min_free_heap), min_free_heap / 1024.0f);
  Serial.printf(" largest_block=%u largest_block_kb=%.2f",
                static_cast<unsigned int>(largest_block), largest_block / 1024.0f);
  Serial.printf(" heap_delta=%ld heap_delta_kb=%.2f\n",
                static_cast<long>(heap_delta), heap_delta / 1024.0f);
}

void updateTimingStats() {
  total_inferences++;
  total_quant_time_us += quant_time_us;
  total_invoke_time_us += invoke_time_us;
  total_dequant_time_us += dequant_time_us;
  total_total_time_us += total_time_us;

  if (quant_time_us < min_quant_time_us) min_quant_time_us = quant_time_us;
  if (quant_time_us > max_quant_time_us) max_quant_time_us = quant_time_us;
  if (invoke_time_us < min_invoke_time_us) min_invoke_time_us = invoke_time_us;
  if (invoke_time_us > max_invoke_time_us) max_invoke_time_us = invoke_time_us;
  if (dequant_time_us < min_dequant_time_us) min_dequant_time_us = dequant_time_us;
  if (dequant_time_us > max_dequant_time_us) max_dequant_time_us = dequant_time_us;
  if (total_time_us < min_total_time_us) min_total_time_us = total_time_us;
  if (total_time_us > max_total_time_us) max_total_time_us = total_time_us;
}

void printTimingSummary() {
  const float avg_quant_us = (total_inferences > 0) ? (static_cast<float>(total_quant_time_us) / total_inferences) : 0.0f;
  const float avg_invoke_us = (total_inferences > 0) ? (static_cast<float>(total_invoke_time_us) / total_inferences) : 0.0f;
  const float avg_dequant_us = (total_inferences > 0) ? (static_cast<float>(total_dequant_time_us) / total_inferences) : 0.0f;
  const float avg_total_us = (total_inferences > 0) ? (static_cast<float>(total_total_time_us) / total_inferences) : 0.0f;
  const float estimated_energy_mj = (avg_total_us / 1000000.0f) * 264.0f;

  Serial.printf("SUMMARY total_inferences=%lu", total_inferences);
  Serial.printf(" quant_avg_us=%.2f quant_min_us=%lu quant_max_us=%lu",
                avg_quant_us,
                static_cast<unsigned long>(min_quant_time_us == UINT32_MAX ? 0 : min_quant_time_us),
                static_cast<unsigned long>(max_quant_time_us));
  Serial.printf(" invoke_avg_us=%.2f invoke_min_us=%lu invoke_max_us=%lu",
                avg_invoke_us,
                static_cast<unsigned long>(min_invoke_time_us == UINT32_MAX ? 0 : min_invoke_time_us),
                static_cast<unsigned long>(max_invoke_time_us));
  Serial.printf(" dequant_avg_us=%.2f dequant_min_us=%lu dequant_max_us=%lu",
                avg_dequant_us,
                static_cast<unsigned long>(min_dequant_time_us == UINT32_MAX ? 0 : min_dequant_time_us),
                static_cast<unsigned long>(max_dequant_time_us));
  Serial.printf(" total_avg_us=%.2f total_min_us=%lu total_max_us=%lu",
                avg_total_us,
                static_cast<unsigned long>(min_total_time_us == UINT32_MAX ? 0 : min_total_time_us),
                static_cast<unsigned long>(max_total_time_us));
  Serial.printf(" estimated_energy_mj=%.6f\n", estimated_energy_mj);
}

bool writeObservationToInputTensor(const float* observation) {
  uint32_t quant_start_us = micros();

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
    Serial.println("ERROR unsupported_model_io");
    return false;
  }

  quant_time_us = micros() - quant_start_us;
  return true;
}

bool readOutputFromTensor(float* frequency, float* amplitude) {
  uint32_t dequant_start_us = micros();

  if (model_io_type == kModelIOFloat32) {
    *frequency = output_tensor->data.f[0];
    *amplitude = output_tensor->data.f[1];
  } else if (model_io_type == kModelIOInt8) {
    *frequency = (static_cast<int32_t>(output_tensor->data.int8[0]) - output_zero_point) * output_scale;
    *amplitude = (static_cast<int32_t>(output_tensor->data.int8[1]) - output_zero_point) * output_scale;
  } else {
    Serial.println("ERROR unsupported_model_io");
    return false;
  }

  dequant_time_us = micros() - dequant_start_us;
  return true;
}

bool runInference(const float* observation, bool verbose) {
  if (input_feature_count <= 0 || input_feature_count > kMaxObservationDim) {
    Serial.printf("ERROR unsupported_input_dim=%d max_supported=%d\n", input_feature_count, kMaxObservationDim);
    return false;
  }
  if (output_feature_count < 2) {
    Serial.println("ERROR expected_at_least_2_outputs");
    return false;
  }

  free_heap_before = esp_get_free_heap_size();

  if (!writeObservationToInputTensor(observation)) {
    return false;
  }

  uint32_t invoke_start_us = micros();
  const TfLiteStatus invoke_status = interpreter->Invoke();
  invoke_time_us = micros() - invoke_start_us;
  free_heap_after = esp_get_free_heap_size();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR invoke_failed");
    return false;
  }

  if (!readOutputFromTensor(&last_frequency, &last_amplitude)) {
    return false;
  }

  total_time_us = quant_time_us + invoke_time_us + dequant_time_us;
  updateTimingStats();

  if (verbose) {
    const float dbs_freq = 185.0f * ((last_frequency + 1.0f) / 2.0f);
    const float dbs_amp = 5000.0f * ((last_amplitude + 1.0f) / 2.0f);
    Serial.printf("RUN_RESULT out0=%.6f out1=%.6f dbs_freq_hz=%.2f dbs_amp=%.2f quant_us=%lu invoke_us=%lu dequant_us=%lu total_us=%lu\n",
                  last_frequency,
                  last_amplitude,
                  dbs_freq,
                  dbs_amp,
                  static_cast<unsigned long>(quant_time_us),
                  static_cast<unsigned long>(invoke_time_us),
                  static_cast<unsigned long>(dequant_time_us),
                  static_cast<unsigned long>(total_time_us));
    printMemoryStats("single_run");
  }

  return true;
}

bool parseObservationCommand(const String& command) {
  char buffer[256];
  command.toCharArray(buffer, sizeof(buffer));

  char* token = strtok(buffer, " ,\t");
  if (token == nullptr) {
    return false;
  }

  int values_read = 0;
  while ((token = strtok(nullptr, " ,\t")) != nullptr) {
    if (values_read >= input_feature_count) {
      Serial.printf("ERROR too_many_values expected=%d\n", input_feature_count);
      return false;
    }
    current_observation[values_read++] = static_cast<float>(atof(token));
  }

  if (values_read != input_feature_count) {
    Serial.printf("ERROR wrong_value_count expected=%d received=%d\n", input_feature_count, values_read);
    return false;
  }

  for (int i = values_read; i < kMaxObservationDim; ++i) {
    current_observation[i] = 0.0f;
  }

  Serial.printf("OBS_OK input_dim=%d\n", input_feature_count);
  return true;
}

void printBenchResult(unsigned long runs) {
  const float avg_quant_us = (total_inferences > 0) ? (static_cast<float>(total_quant_time_us) / total_inferences) : 0.0f;
  const float avg_invoke_us = (total_inferences > 0) ? (static_cast<float>(total_invoke_time_us) / total_inferences) : 0.0f;
  const float avg_dequant_us = (total_inferences > 0) ? (static_cast<float>(total_dequant_time_us) / total_inferences) : 0.0f;
  const float avg_total_us = (total_inferences > 0) ? (static_cast<float>(total_total_time_us) / total_inferences) : 0.0f;

  Serial.printf(
      "BENCH_RESULT runs=%lu model_io=%s input_dim=%d output_dim=%d quant_avg_us=%.2f invoke_avg_us=%.2f "
      "dequant_avg_us=%.2f total_avg_us=%.2f min_invoke_us=%lu max_invoke_us=%lu\n",
      runs,
      modelIOTypeName(),
      input_feature_count,
      output_feature_count,
      avg_quant_us,
      avg_invoke_us,
      avg_dequant_us,
      avg_total_us,
      static_cast<unsigned long>(min_invoke_time_us == UINT32_MAX ? 0 : min_invoke_time_us),
      static_cast<unsigned long>(max_invoke_time_us));
}

void runBenchmark(unsigned long runs) {
  if (runs == 0) {
    Serial.println("ERROR benchmark_count_must_be_positive");
    return;
  }

  resetTimingStats();
  for (unsigned long i = 0; i < runs; ++i) {
    if (!runInference(current_observation, false)) {
      Serial.printf("ERROR benchmark_aborted iteration=%lu\n", i);
      return;
    }
  }

  printBenchResult(runs);
  Serial.println("BENCH_DONE");
  Serial.flush();
}

void handleSerialCommand(String command) {
  command.trim();
  if (command.length() == 0) {
    return;
  }

  if (command.equalsIgnoreCase("help")) {
    printHelp();
    return;
  }

  if (command.equalsIgnoreCase("info")) {
    printModelInfo();
    return;
  }

  if (command.equalsIgnoreCase("defaults")) {
    setDefaultObservation();
    Serial.printf("DEFAULTS_OK input_dim=%d\n", input_feature_count);
    return;
  }

  if (command.startsWith("sample") || command.startsWith("SAMPLE")) {
    parseObservationCommand(command);
    return;
  }

  if (command.equalsIgnoreCase("run")) {
    resetTimingStats();
    if (runInference(current_observation, true)) {
      printTimingSummary();
    }
    return;
  }

  if (command.startsWith("bench") || command.startsWith("BENCH")) {
    const int split = command.indexOf(' ');
    unsigned long runs = 200;
    if (split > 0) {
      runs = static_cast<unsigned long>(command.substring(split + 1).toInt());
    }
    runBenchmark(runs);
    return;
  }

  Serial.print("ERROR unknown_command=");
  Serial.println(command);
}

void printReadyLine() {
  Serial.printf("READY runtime=arduino library=chirale_tflm model_io=%s input_dim=%d output_dim=%d tensor_arena_bytes=%d\n",
                modelIOTypeName(),
                input_feature_count,
                output_feature_count,
                kTensorArenaSize);
  Serial.flush();
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  unsigned long serial_wait_start = millis();
  while (!Serial && (millis() - serial_wait_start < kSerialAttachWaitMs)) {
    delay(10);
  }
  delay(kSerialSettleDelayMs);

  tflite_model = tflite::GetModel(dbs_model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR schema_version=%d expected=%d\n",
                  tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (!registerModelOps()) {
    return;
  }

  if (!psramFound()) {
    Serial.println("ERROR psram_not_found");
    return;
  }

  tensor_arena = static_cast<uint8_t*>(
      heap_caps_aligned_alloc(kTensorArenaAlignment, kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (tensor_arena == nullptr) {
    Serial.printf("ERROR tensor_arena_alloc_failed bytes=%d\n", kTensorArenaSize);
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR allocate_tensors_failed");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  input_feature_count = input_tensor->dims->data[input_tensor->dims->size - 1];
  output_feature_count = output_tensor->dims->data[output_tensor->dims->size - 1];

  if (input_feature_count <= 0 || input_feature_count > kMaxObservationDim) {
    Serial.printf("ERROR input_dim=%d max_supported=%d\n",
                  input_feature_count, kMaxObservationDim);
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
  }

  setDefaultObservation();
  resetTimingStats();

  printReadyLine();
}

void loop() {
  if (Serial.available() > 0) {
    const String command = Serial.readStringUntil('\n');
    handleSerialCommand(command);
  }
  delay(10);
}
