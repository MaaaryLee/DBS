/*
 * ESP32 DBS Inference with Power & Memory Profiling
 * 
 * This sketch:
 * 1. Loads the TFLite model from model.h
 * 2. Runs inference on sensor data
 * 3. Measures power consumption
 * 4. Tracks memory usage
 * 5. Outputs DBS parameters (frequency, amplitude)
 */

#include "model.h"  // Your TFLite model as C byte array
#include <math.h>   // lrintf

// For TensorFlow Lite Micro
// You'll need to install: Arduino TensorFlow Lite library
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// For power measurement (if using external power monitor)
#include <driver/adc.h>
#include <esp_adc_cal.h>

// For memory tracking
#include <esp_heap_caps.h>

// Model and interpreter
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Memory for model (adjust size if needed)
constexpr int kTensorArenaSize = 10 * 1024;  // 10KB - adjust based on model
uint8_t tensor_arena[kTensorArenaSize];

// Power measurement variables
unsigned long inference_start_time = 0;
unsigned long inference_end_time = 0;
float inference_time_ms = 0.0;

// Memory tracking
size_t free_heap_before = 0;
size_t free_heap_after = 0;
size_t min_free_heap = 0;

// Statistics
unsigned long total_inferences = 0;
float total_power_time = 0.0;
float max_inference_time = 0.0;
float min_inference_time = 9999.0;

// Model IO detection + quantization parameters (for INT8 models)
enum ModelIOType {
  kModelIOFloat32 = 0,
  kModelIOInt8 = 1,
  kModelIOUnsupported = 2,
};

ModelIOType model_io_type = kModelIOUnsupported;

float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;

// Separate timing (microseconds)
uint32_t quant_time_us = 0;
uint32_t invoke_time_us = 0;
uint32_t dequant_time_us = 0;
uint32_t total_time_us = 0;

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

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== ESP32 DBS Inference System ===");
  Serial.println("Initializing TensorFlow Lite...");
  
  // Load model from model.h
  // model.h defines:
  // - `unsigned char model[]`
  // - `unsigned int model_len`
  model = tflite::GetModel(model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema version %d not supported. Supported is %d.\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  
  Serial.println("Model loaded successfully!");
  Serial.printf("Model size: %d bytes\n", model_len);
  
  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }
  
  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Model initialized!");
  Serial.printf("Input shape: [%d, %d]\n", input->dims->data[0], input->dims->data[1]);
  Serial.printf("Output shape: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);

  // Detect model IO type
  Serial.printf("Input tensor type: %d\n", static_cast<int>(input->type));
  Serial.printf("Output tensor type: %d\n", static_cast<int>(output->type));

  if (input->type == kTfLiteFloat32 && output->type == kTfLiteFloat32) {
    model_io_type = kModelIOFloat32;
    Serial.println("Detected model IO: FP32 (float32 input/output)");
  } else if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    model_io_type = kModelIOInt8;
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;
    output_scale = output->params.scale;
    output_zero_point = output->params.zero_point;
    Serial.println("Detected model IO: INT8 (int8 input/output)");
    Serial.printf("Input quant params: scale=%.10f, zero_point=%d\n", input_scale, input_zero_point);
    Serial.printf("Output quant params: scale=%.10f, zero_point=%d\n", output_scale, output_zero_point);
  } else {
    model_io_type = kModelIOUnsupported;
    Serial.println("Detected model IO: UNSUPPORTED (mixed or unknown tensor types)");
    Serial.println("Expected either float32->float32 or int8->int8.");
    Serial.println("Fix by embedding a matching model in model.h.");
  }
  
  // Print initial memory stats
  printMemoryStats("Initial");
  
  Serial.println("\n=== Ready for inference ===\n");
}

void loop() {
  // Simulate sensor data (replace with actual sensor readings)
  // Input: 4-element observation vector (Hjorth parameters)
  float observation[4] = {
    0.5,  // Example: Hjorth parameter 1
    0.3,  // Example: Hjorth parameter 2
    0.7,  // Example: Hjorth parameter 3
    0.2   // Example: Hjorth parameter 4
  };
  
  // Measure memory before inference
  free_heap_before = esp_get_free_heap_size();
  min_free_heap = esp_get_minimum_free_heap_size();
  
  // Prepare input (and measure quantization time separately)
  uint32_t quant_start_us = micros();
  if (model_io_type == kModelIOFloat32) {
    for (int i = 0; i < 4; i++) {
      input->data.f[i] = observation[i];
    }
  } else if (model_io_type == kModelIOInt8) {
    // Quantize: int8 = round(float / scale + zero_point)
    // Clamp to int8 range [-128, 127]
    for (int i = 0; i < 4; i++) {
      int32_t q = static_cast<int32_t>(lrintf(observation[i] / input_scale)) + input_zero_point;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input->data.int8[i] = static_cast<int8_t>(q);
    }
  }
  quant_time_us = micros() - quant_start_us;

  // Measure inference time (Invoke only)
  uint32_t invoke_start_us = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  invoke_time_us = micros() - invoke_start_us;
  inference_time_ms = invoke_time_us / 1000.0f;
  
  // Measure memory after inference
  free_heap_after = esp_get_free_heap_size();
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke() failed!");
    return;
  }
  
  // Read output (and measure dequantization time separately)
  float frequency = 0.0f;
  float amplitude = 0.0f;
  uint32_t dequant_start_us = micros();
  if (model_io_type == kModelIOFloat32) {
    frequency = output->data.f[0];
    amplitude = output->data.f[1];
  } else if (model_io_type == kModelIOInt8) {
    // Dequantize: float = (int8 - zero_point) * scale
    frequency = (static_cast<int32_t>(output->data.int8[0]) - output_zero_point) * output_scale;
    amplitude = (static_cast<int32_t>(output->data.int8[1]) - output_zero_point) * output_scale;
  }
  dequant_time_us = micros() - dequant_start_us;

  total_time_us = quant_time_us + invoke_time_us + dequant_time_us;
  
  // Convert normalized actions to actual DBS parameters
  // Assuming actions are in [-1, 1], convert to [0, 185] Hz and [0, 5000] mA
  float dbs_freq = 185 * ((frequency + 1.0) / 2.0);
  float dbs_amp = 5000 * ((amplitude + 1.0) / 2.0);
  
  // Update statistics
  total_inferences++;
  total_power_time += inference_time_ms;
  if (inference_time_ms > max_inference_time) max_inference_time = inference_time_ms;
  if (inference_time_ms < min_inference_time) min_inference_time = inference_time_ms;

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
  
  // Print results every 10 inferences
  if (total_inferences % 10 == 0) {
    printInferenceResults(dbs_freq, dbs_amp);
    printMemoryStats("After inference");
    printPowerStats();
  }
  
  delay(100);  // Adjust based on your sampling rate
}

void printInferenceResults(float freq, float amp) {
  Serial.println("\n--- Inference Results ---");
  Serial.printf("DBS Frequency: %.2f Hz\n", freq);
  Serial.printf("DBS Amplitude: %.2f mA\n", amp);
  Serial.printf("Inference time: %.3f ms\n", inference_time_ms);
  Serial.printf("Quantize: %lu us | Invoke: %lu us | Dequant: %lu us | Total: %lu us\n",
                static_cast<unsigned long>(quant_time_us),
                static_cast<unsigned long>(invoke_time_us),
                static_cast<unsigned long>(dequant_time_us),
                static_cast<unsigned long>(total_time_us));
}

void printMemoryStats(const char* label) {
  Serial.printf("\n--- Memory Stats (%s) ---\n", label);
  Serial.printf("Free heap: %d bytes (%.2f KB)\n", 
                esp_get_free_heap_size(), 
                esp_get_free_heap_size() / 1024.0);
  Serial.printf("Min free heap: %d bytes (%.2f KB)\n", 
                esp_get_minimum_free_heap_size(),
                esp_get_minimum_free_heap_size() / 1024.0);
  Serial.printf("Largest free block: %d bytes (%.2f KB)\n",
                esp_get_largest_free_block(),
                esp_get_largest_free_block() / 1024.0);
  
  // Memory used by inference
  // Use a signed delta to avoid underflow if heap increases.
  int32_t memory_delta = static_cast<int32_t>(free_heap_before) - static_cast<int32_t>(free_heap_after);
  Serial.printf("Heap delta (before - after): %ld bytes (%.2f KB)\n",
                static_cast<long>(memory_delta), memory_delta / 1024.0f);
  
  // Tensor arena usage
  Serial.printf("Tensor arena size: %d bytes (%.2f KB)\n",
                kTensorArenaSize, kTensorArenaSize / 1024.0);
}

void printPowerStats() {
  Serial.println("\n--- Power/Performance Stats ---");
  Serial.printf("Total inferences: %lu\n", total_inferences);
  Serial.printf("Average inference time: %.3f ms\n", total_power_time / total_inferences);
  Serial.printf("Min inference time: %.3f ms\n", min_inference_time);
  Serial.printf("Max inference time: %.3f ms\n", max_inference_time);

  // Separate timing stats
  const float avg_quant_us = (total_inferences > 0) ? (static_cast<float>(total_quant_time_us) / total_inferences) : 0.0f;
  const float avg_invoke_us = (total_inferences > 0) ? (static_cast<float>(total_invoke_time_us) / total_inferences) : 0.0f;
  const float avg_dequant_us = (total_inferences > 0) ? (static_cast<float>(total_dequant_time_us) / total_inferences) : 0.0f;
  const float avg_total_us = (total_inferences > 0) ? (static_cast<float>(total_total_time_us) / total_inferences) : 0.0f;

  Serial.printf("Model IO mode: %s\n",
                (model_io_type == kModelIOFloat32) ? "FP32" :
                (model_io_type == kModelIOInt8) ? "INT8" : "UNSUPPORTED");

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
  
  // Estimate power (rough calculation)
  // ESP32 active current ~80mA @ 3.3V = ~264mW
  // Inference time correlates with energy consumption
  const float assumed_power_mw = 264.0f;
  float avg_energy_per_inference = (inference_time_ms / 1000.0f) * assumed_power_mw;  // mJ
  Serial.printf("Estimated energy per inference: %.3f mJ\n", avg_energy_per_inference);
  Serial.printf("Assumed power during inference: %.2f mW\n", assumed_power_mw);
}



