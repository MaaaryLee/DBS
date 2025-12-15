#include <ArduTFLite.h>
#include "model.h"
#include <Arduino.h>
extern "C" {
#include "esp32s3/rom/cache.h"
}

constexpr int kDbsTensorArenaSize = 16 * 1024;
alignas(16) uint8_t dbs_tensor_arena[kDbsTensorArenaSize];
double d0M = 0;
double d0S = 0;
double d1M = 0;
double d1S = 0;
double d2M = 0;
double d2S = 0;
double d0_0 = 0;
double d0_1 = 0;
double d0_2 = 0;
double d0OldM;
double d1OldM;
double d2OldM;
int numReadings = 0;

IRAM_ATTR void invalidate_cache() {
    Cache_Invalidate_ICache_All();
}

void setup() {
  if (!modelInit(model, dbs_tensor_arena, kDbsTensorArenaSize)){
      Serial.println("percom_network_tflite initialization failed!");
      while(true);
    }

    Serial.begin(115200);

}

void loop() {
  invalidate_cache();
  for (int i=0; i<250; i++){
      d0OldM = d0M;
      d1OldM = d1M;
      d2OldM = d2M;
      d0_0 = d0_1;
      d0_1 = d0_2;
      d0_2 = random(-80000, 80000)/1000.0;
      numReadings++;
      double d1 = d0_1 - d0_0;
      double d2 = (d0_2-d0_1) - (d0_1 - d0_0);

      d0M = d0M + (d0_2 - d0M)/numReadings;
      d0S = d0S + (d0_2 - d0M)*(d0_2 - d0OldM);
      d1M = d1M + (d1 - d0M)/numReadings;
      d1S = d1S + (d1 - d0M)*(d1 - d1OldM);
      d2M = d2M + (d2 - d0M)/numReadings;
      d2S = d2S + (d2 - d0M)*(d2 - d2OldM);
    }

    double A = d0S/250; // A is now var(x)
    double sd = sqrt(A);
    double M = d1S/250; // M is now var(x')
    double C = d2S/250; // C is now var(x'')
    C = sqrt(C)/M; // C is now complexity
    M = sqrt(M)/A; // M is now mobility

    // Normalize 
    A = (A-672.2282360031815)/(1047.66991012089-672.2282360031815);
    M = (M-0.09201175246694249)/(0.09689537936008763-0.09201175246694249);
    C = (C-0.7911088367808019)/(0.8618411558467868-0.7911088367808019);
    sd = (sd-25.849944778332663)/(32.36659864207871-25.849944778332663);

    modelSetInput(sd,0);
    modelSetInput(A,1);
    modelSetInput(M,2);
    modelSetInput(C,3);

    int start = micros();
    if(!modelRunInference()){
      Serial.println("RunInference Failed!");
      return;
    }

    Serial.println(micros()-start); // test inference time


    double freq = modelGetOutput(0); // freq
    double amp = modelGetOutput(1); // amp

    // Serial.print("freq: ");
    // Serial.println(freq);
    // Serial.print("amp: ");
    // Serial.println(amp);

    d0M = 0;
    d0S = 0;
    d1M = 0;
    d1S = 0;
    d2M = 0;
    d2S = 0;
    numReadings = 0;

}