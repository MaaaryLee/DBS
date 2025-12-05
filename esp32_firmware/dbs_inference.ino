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

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n=== ESP32 DBS Inference System ===");
  Serial.println("Initializing TensorFlow Lite...");
  
  // Load model from model.h
  model = tflite::GetModel(model_data);
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
  
  // Prepare input
  for (int i = 0; i < 4; i++) {
    input->data.f[i] = observation[i];
  }
  
  // Measure inference time (proxy for power)
  inference_start_time = micros();
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  inference_end_time = micros();
  inference_time_ms = (inference_end_time - inference_start_time) / 1000.0;
  
  // Measure memory after inference
  free_heap_after = esp_get_free_heap_size();
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke() failed!");
    return;
  }
  
  // Get output (DBS parameters)
  float frequency = output->data.f[0];
  float amplitude = output->data.f[1];
  
  // Convert normalized actions to actual DBS parameters
  // Assuming actions are in [-1, 1], convert to [0, 185] Hz and [0, 5000] mA
  float dbs_freq = 185 * ((frequency + 1.0) / 2.0);
  float dbs_amp = 5000 * ((amplitude + 1.0) / 2.0);
  
  // Update statistics
  total_inferences++;
  total_power_time += inference_time_ms;
  if (inference_time_ms > max_inference_time) max_inference_time = inference_time_ms;
  if (inference_time_ms < min_inference_time) min_inference_time = inference_time_ms;
  
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
  size_t memory_used = free_heap_before - free_heap_after;
  Serial.printf("Memory used by inference: %d bytes (%.2f KB)\n",
                memory_used, memory_used / 1024.0);
  
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
  
  // Estimate power (rough calculation)
  // ESP32 active current ~80mA @ 3.3V = ~264mW
  // Inference time correlates with energy consumption
  float avg_energy_per_inference = (inference_time_ms / 1000.0) * 264.0;  // mJ
  Serial.printf("Estimated energy per inference: %.3f mJ\n", avg_energy_per_inference);
  Serial.printf("Estimated power during inference: %.2f mW\n", 
                264.0 * (inference_time_ms / 1000.0) / (inference_time_ms / 1000.0));
}



