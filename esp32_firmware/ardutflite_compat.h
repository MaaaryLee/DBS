#pragma once

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <stdio.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace dbs_ardutflite_compat {

static tflite::AllOpsResolver tfl_ops_resolver;
static const tflite::Model* tfl_model = nullptr;
static tflite::MicroInterpreter* tfl_interpreter = nullptr;
static TfLiteTensor* tfl_input_tensor = nullptr;
static TfLiteTensor* tfl_output_tensor = nullptr;
static char tfl_last_error[128] = "not initialized";

inline void setLastError(const char* message) {
  snprintf(tfl_last_error, sizeof(tfl_last_error), "%s", message);
}

inline void setIndexedError(const char* prefix, int index, int limit) {
  snprintf(tfl_last_error, sizeof(tfl_last_error), "%s (index=%d limit=%d)", prefix, index, limit);
}

inline int flatTensorLength(TfLiteTensor* tensor) {
  if (tensor == nullptr) return -1;
  if (tensor->type == kTfLiteFloat32) return tensor->bytes / static_cast<int>(sizeof(float));
  if (tensor->type == kTfLiteInt8) return tensor->bytes / static_cast<int>(sizeof(int8_t));
  return -1;
}

}  // namespace dbs_ardutflite_compat

inline bool modelInit(const unsigned char* model, byte* tensorArena, int tensorArenaSize) {
  using namespace dbs_ardutflite_compat;

  setLastError("OK");
  tfl_input_tensor = nullptr;
  tfl_output_tensor = nullptr;

  if (model == nullptr) {
    setLastError("Model pointer is null.");
    return false;
  }
  if (tensorArena == nullptr || tensorArenaSize <= 0) {
    setLastError("Tensor arena is invalid.");
    return false;
  }

  if (tfl_interpreter != nullptr) {
    delete tfl_interpreter;
    tfl_interpreter = nullptr;
  }

  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    setLastError("Model schema version mismatch.");
    return false;
  }

  tfl_interpreter = new tflite::MicroInterpreter(tfl_model, tfl_ops_resolver, tensorArena, tensorArenaSize);
  if (tfl_interpreter == nullptr) {
    setLastError("Interpreter allocation failed.");
    return false;
  }
  if (tfl_interpreter->AllocateTensors() != kTfLiteOk) {
    setLastError("AllocateTensors failed (arena too small or unsupported op).");
    delete tfl_interpreter;
    tfl_interpreter = nullptr;
    return false;
  }

  tfl_input_tensor = tfl_interpreter->input(0);
  tfl_output_tensor = tfl_interpreter->output(0);
  if (tfl_input_tensor == nullptr || tfl_output_tensor == nullptr) {
    setLastError("Input or output tensor is null after initialization.");
    delete tfl_interpreter;
    tfl_interpreter = nullptr;
    tfl_input_tensor = nullptr;
    tfl_output_tensor = nullptr;
    return false;
  }

  return true;
}

inline bool modelSetInput(float inputValue, int index) {
  using namespace dbs_ardutflite_compat;

  if (tfl_input_tensor == nullptr) {
    setLastError("Input tensor unavailable.");
    return false;
  }

  if (tfl_input_tensor->type == kTfLiteFloat32) {
    int n = tfl_input_tensor->bytes / static_cast<int>(sizeof(float));
    if (index < 0 || index >= n) {
      setIndexedError("Input tensor index out of range.", index, n);
      return false;
    }
    tfl_input_tensor->data.f[index] = inputValue;
    setLastError("OK");
    return true;
  }

  if (tfl_input_tensor->type == kTfLiteInt8) {
    int n = tfl_input_tensor->bytes / static_cast<int>(sizeof(int8_t));
    if (index < 0 || index >= n) {
      setIndexedError("Input tensor index out of range.", index, n);
      return false;
    }

    const float scale = tfl_input_tensor->params.scale;
    const int zp = tfl_input_tensor->params.zero_point;
    int32_t q = static_cast<int32_t>(lrintf(inputValue / scale)) + zp;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    tfl_input_tensor->data.int8[index] = static_cast<int8_t>(q);
    setLastError("OK");
    return true;
  }

  setLastError("Unsupported input tensor type.");
  return false;
}

inline bool modelRunInference() {
  using namespace dbs_ardutflite_compat;

  if (tfl_interpreter == nullptr) {
    setLastError("Interpreter unavailable.");
    return false;
  }
  if (tfl_interpreter->Invoke() != kTfLiteOk) {
    setLastError("Invoke failed.");
    return false;
  }
  setLastError("OK");
  return true;
}

inline float modelGetOutput(int index) {
  using namespace dbs_ardutflite_compat;

  if (tfl_output_tensor == nullptr) {
    setLastError("Output tensor unavailable.");
    return -1.0f;
  }

  if (tfl_output_tensor->type == kTfLiteFloat32) {
    int n = tfl_output_tensor->bytes / static_cast<int>(sizeof(float));
    if (index < 0 || index >= n) {
      setIndexedError("Output tensor index out of range.", index, n);
      return -1.0f;
    }
    setLastError("OK");
    return tfl_output_tensor->data.f[index];
  }

  if (tfl_output_tensor->type == kTfLiteInt8) {
    int n = tfl_output_tensor->bytes / static_cast<int>(sizeof(int8_t));
    if (index < 0 || index >= n) {
      setIndexedError("Output tensor index out of range.", index, n);
      return -1.0f;
    }

    const float scale = tfl_output_tensor->params.scale;
    const int zp = tfl_output_tensor->params.zero_point;
    int8_t q = tfl_output_tensor->data.int8[index];
    setLastError("OK");
    return (static_cast<int32_t>(q) - zp) * scale;
  }

  setLastError("Unsupported output tensor type.");
  return -1.0f;
}

inline int modelGetInputLength() {
  return dbs_ardutflite_compat::flatTensorLength(dbs_ardutflite_compat::tfl_input_tensor);
}

inline int modelGetOutputLength() {
  return dbs_ardutflite_compat::flatTensorLength(dbs_ardutflite_compat::tfl_output_tensor);
}

inline bool modelInputIsInt8() {
  return dbs_ardutflite_compat::tfl_input_tensor != nullptr &&
         dbs_ardutflite_compat::tfl_input_tensor->type == kTfLiteInt8;
}

inline bool modelOutputIsInt8() {
  return dbs_ardutflite_compat::tfl_output_tensor != nullptr &&
         dbs_ardutflite_compat::tfl_output_tensor->type == kTfLiteInt8;
}

inline const char* modelGetLastError() {
  return dbs_ardutflite_compat::tfl_last_error;
}
