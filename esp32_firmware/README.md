# ESP32 DBS Inference Firmware

This directory contains the ESP32 firmware for running your quantized DBS model and measuring power/memory usage.

## Files

- `dbs_inference.ino` - Main Arduino sketch with inference and profiling
- `model.h` - Your TFLite model (copy from root directory)

## Setup Instructions

### 1. Install Required Libraries

In Arduino IDE:
1. Go to **Tools → Manage Libraries**
2. Install:
   - **TensorFlowLite_ESP32** (by TensorFlow team)
   - **ESP32** board support (if not already installed)

### 2. Board Configuration

1. Go to **Tools → Board → ESP32 Arduino**
2. Select your ESP32 board (e.g., "ESP32 Dev Module")
3. Set:
   - **Upload Speed**: 115200
   - **CPU Frequency**: 240MHz (for best performance)
   - **Flash Frequency**: 80MHz
   - **Partition Scheme**: "Default 4MB with spiffs"

### 3. Copy Model File

```bash
# From project root
cp model.h esp32_firmware/
```

### 4. Open in Arduino IDE

1. Open `dbs_inference.ino` in Arduino IDE
2. Verify the code compiles (Ctrl+R / Cmd+R)
3. Connect your ESP32 via USB
4. Upload (Ctrl+U / Cmd+U)

## Using Cursor Instead of Arduino IDE

You can write/edit the code in Cursor, then use Arduino CLI to compile and upload:

### Install Arduino CLI

```bash
# macOS
brew install arduino-cli

# Or download from: https://arduino.github.io/arduino-cli/
```

### Setup Arduino CLI

```bash
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "TensorFlowLite_ESP32"
```

### Compile and Upload from Cursor

```bash
# From esp32_firmware directory
arduino-cli compile --fqbn esp32:esp32:esp32 .
arduino-cli upload -p /dev/cu.usbserial-* --fqbn esp32:esp32:esp32 .
```

## What the Code Does

1. **Loads Model**: Reads TFLite model from `model.h`
2. **Runs Inference**: Takes 4-element observation, outputs 2-element action
3. **Measures Memory**: Tracks heap usage before/after inference
4. **Measures Performance**: Tracks inference time (proxy for power)
5. **Outputs Results**: Prints DBS frequency/amplitude via Serial

## Power Measurement

The code measures:
- **Inference time** (correlates with power consumption)
- **Memory usage** (heap before/after)
- **Energy estimation** (based on ESP32 power specs)

For **actual power measurement**, you'll need:
- External power monitor (e.g., INA219, INA260)
- Or use ESP32's built-in ADC to measure current (requires shunt resistor)

## Memory Profiling

The code tracks:
- Free heap before/after inference
- Minimum free heap (lowest point)
- Largest free block
- Tensor arena usage

## Serial Output

Connect to Serial Monitor (115200 baud) to see:
- Model initialization status
- Memory statistics
- Inference results (every 10 inferences)
- Power/performance statistics

## Next Steps

1. **Test inference**: Verify model runs correctly
2. **Add real sensor data**: Replace simulated observation with actual sensor readings
3. **Add power monitor**: Integrate INA219 or similar for actual power measurement
4. **Compare FP32 vs INT8**: Deploy both versions and compare power/memory

## Troubleshooting

### "Model schema version not supported"
- Update TensorFlow Lite library to latest version

### "AllocateTensors() failed"
- Increase `kTensorArenaSize` in the code (currently 10KB)
- Model might need more memory

### "Out of memory"
- Reduce `kTensorArenaSize` if too large
- Check partition scheme (use larger flash if needed)

### Model not found
- Ensure `model.h` is in the same directory as `.ino` file
- Check that `model.h` contains valid TFLite model data



