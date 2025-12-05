/*
 * ESP32 Power Monitor with INA219
 * 
 * For actual power measurement, use an INA219 current/power monitor
 * This is optional but provides real power consumption data
 */

// Uncomment if you have INA219 connected
// #define USE_INA219

#ifdef USE_INA219
#include <Wire.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

void setupPowerMonitor() {
  if (!ina219.begin()) {
    Serial.println("Failed to find INA219 chip");
    while (1) { delay(10); }
  }
  Serial.println("INA219 Power Monitor initialized");
}

float measurePower() {
  float shuntvoltage = ina219.getShuntVoltage_mV();
  float busvoltage = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_mW = ina219.getPower_mW();
  float loadvoltage = busvoltage + (shuntvoltage / 1000);
  
  Serial.printf("Bus Voltage:   %.3f V\n", busvoltage);
  Serial.printf("Shunt Voltage: %.3f mV\n", shuntvoltage);
  Serial.printf("Load Voltage:  %.3f V\n", loadvoltage);
  Serial.printf("Current:       %.3f mA\n", current_mA);
  Serial.printf("Power:         %.3f mW\n", power_mW);
  
  return power_mW;
}
#else
void setupPowerMonitor() {
  Serial.println("INA219 not configured - using time-based estimation");
}

float measurePower() {
  // Return estimated power based on inference time
  // ESP32 active: ~80mA @ 3.3V = ~264mW
  return 264.0;
}
#endif



