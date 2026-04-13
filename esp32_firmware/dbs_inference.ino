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

#include <esp_heap_caps.h>

namespace {

constexpr int kTensorArenaSize = 16 * 1024;
constexpr int kMaxObservationDim = 8;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

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
  Serial.print("Current observation: [");
  for (int i = 0; i < input_feature_count; ++i) {
    Serial.print(current_observation[i], 6);
    if (i + 1 < input_feature_count) {
      Serial.print(", ");
    }
  }
  Serial.println("]");
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
  Serial.println("\nCommands:");
  Serial.println("  help");
  Serial.println("  info");
  Serial.println("  defaults");
  Serial.println("  sample <v1> <v2> ... <vN>");
  Serial.println("  run");
  Serial.println("  bench <count>");
}

void printModelInfo() {
  Serial.println("\n--- Model Info ---");
  Serial.printf("Embedded model size: %u bytes\n", dbs_model_len);
  Serial.printf("Input shape: [%d, %d]\n",
                input_tensor->dims->data[0], input_tensor->dims->data[input_tensor->dims->size - 1]);
  Serial.printf("Output shape: [%d, %d]\n",
                output_tensor->dims->data[0], output_tensor->dims->data[output_tensor->dims->size - 1]);
  Serial.printf("Input tensor type: %d\n", static_cast<int>(input_tensor->type));
  Serial.printf("Output tensor type: %d\n", static_cast<int>(output_tensor->type));
  Serial.printf("Model IO mode: %s\n", modelIOTypeName());
  if (model_io_type == kModelIOInt8) {
    Serial.printf("Input quant params : scale=%.10f zero_point=%d\n", input_scale, input_zero_point);
    Serial.printf("Output quant params: scale=%.10f zero_point=%d\n", output_scale, output_zero_point);
  }
  Serial.printf("Tensor arena size: %d bytes\n", kTensorArenaSize);
  printCurrentObservation();
}

void printMemoryStats(const char* label) {
  const size_t free_heap = esp_get_free_heap_size();
  const size_t min_free_heap = esp_get_minimum_free_heap_size();
  const size_t largest_block = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  const int32_t heap_delta = static_cast<int32_t>(free_heap_before) - static_cast<int32_t>(free_heap_after);

  Serial.printf("\n--- Memory Stats (%s) ---\n", label);
  Serial.printf("Free heap: %u bytes (%.2f KB)\n",
                static_cast<unsigned int>(free_heap), free_heap / 1024.0f);
  Serial.printf("Min free heap: %u bytes (%.2f KB)\n",
                static_cast<unsigned int>(min_free_heap), min_free_heap / 1024.0f);
  Serial.printf("Largest free block: %u bytes (%.2f KB)\n",
                static_cast<unsigned int>(largest_block), largest_block / 1024.0f);
  Serial.printf("Heap delta (before - after): %ld bytes (%.2f KB)\n",
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

  Serial.println("\n--- Timing Summary ---");
  Serial.printf("Total inferences: %lu\n", total_inferences);
  Serial.printf("Quantize avg: %.2f us (min=%lu, max=%lu)\n",
                avg_quant_us,
                static_cast<unsigned long>(min_quant_time_us == UINT32_MAX ? 0 : min_quant_time_us),
                static_cast<unsigned long>(max_quant_time_us));
  Serial.printf("Invoke   avg: %.2f us (min=%lu, max=%lu)\n",
                avg_invoke_us,
                static_cast<unsigned long>(min_invoke_time_us == UINT32_MAX ? 0 : min_invoke_time_us),
                static_cast<unsigned long>(max_invoke_time_us));
  Serial.printf("Dequant  avg: %.2f us (min=%lu, max=%lu)\n",
                avg_dequant_us,
                static_cast<unsigned long>(min_dequant_time_us == UINT32_MAX ? 0 : min_dequant_time_us),
                static_cast<unsigned long>(max_dequant_time_us));
  Serial.printf("Total    avg: %.2f us (min=%lu, max=%lu)\n",
                avg_total_us,
                static_cast<unsigned long>(min_total_time_us == UINT32_MAX ? 0 : min_total_time_us),
                static_cast<unsigned long>(max_total_time_us));
  Serial.printf("Estimated energy per inference: %.6f mJ\n", estimated_energy_mj);
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
    Serial.println("Unsupported model IO mode.");
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
    Serial.println("Unsupported model IO mode.");
    return false;
  }

  dequant_time_us = micros() - dequant_start_us;
  return true;
}

bool runInference(const float* observation, bool verbose) {
  if (input_feature_count <= 0 || input_feature_count > kMaxObservationDim) {
    Serial.printf("Input dimension %d is unsupported by this sketch.\n", input_feature_count);
    return false;
  }
  if (output_feature_count < 2) {
    Serial.println("Expected at least 2 output features.");
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
    Serial.println("Invoke() failed!");
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
    Serial.println("\n--- Inference Result ---");
    Serial.printf("Frequency action: %.6f -> DBS %.2f Hz\n", last_frequency, dbs_freq);
    Serial.printf("Amplitude action: %.6f -> DBS %.2f mA\n", last_amplitude, dbs_amp);
    Serial.printf("Quantize=%lu us Invoke=%lu us Dequant=%lu us Total=%lu us\n",
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
      Serial.printf("Too many values. Model expects %d features.\n", input_feature_count);
      return false;
    }
    current_observation[values_read++] = static_cast<float>(atof(token));
  }

  if (values_read != input_feature_count) {
    Serial.printf("Expected %d values, received %d.\n", input_feature_count, values_read);
    return false;
  }

  for (int i = values_read; i < kMaxObservationDim; ++i) {
    current_observation[i] = 0.0f;
  }

  Serial.println("Observation updated.");
  printCurrentObservation();
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
    Serial.println("Benchmark count must be > 0.");
    return;
  }

  resetTimingStats();
  for (unsigned long i = 0; i < runs; ++i) {
    if (!runInference(current_observation, false)) {
      Serial.printf("Benchmark aborted at iteration %lu.\n", i);
      return;
    }
  }

  printTimingSummary();
  printBenchResult(runs);
  Serial.println("BENCH_DONE");
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
    Serial.println("Default observation restored.");
    printCurrentObservation();
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

  Serial.print("Unknown command: ");
  Serial.println(command);
  printHelp();
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  delay(1000);

  Serial.println("\n=== ESP32 DBS Inference Benchmark ===");
  Serial.println("Initializing TensorFlow Lite Micro...");

  tflite_model = tflite::GetModel(dbs_model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema version %d not supported. Supported is %d.\n",
                  tflite_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (!registerModelOps()) {
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  input_feature_count = input_tensor->dims->data[input_tensor->dims->size - 1];
  output_feature_count = output_tensor->dims->data[output_tensor->dims->size - 1];

  if (input_feature_count <= 0 || input_feature_count > kMaxObservationDim) {
    Serial.printf("Input dimension %d exceeds sketch limit (%d).\n",
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

  printModelInfo();
  printHelp();
  Serial.println("\nReady. Send `run` or `bench 200`.");
}

void loop() {
  if (Serial.available() > 0) {
    const String command = Serial.readStringUntil('\n');
    handleSerialCommand(command);
  }
  delay(10);
}
