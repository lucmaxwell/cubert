#include <Arduino.h>
#include <stdio.h>

// put function declarations here:
char msg[256];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  // psramInit();

  // log_d("Total heap: %d", ESP.getHeapSize());
  // log_d("Free heap: %d", ESP.getFreeHeap());
  // log_d("Total PSRAM: %d", ESP.getPsramSize());
  // log_d("Free PSRAM: %d", ESP.getFreePsram());
}

void loop() {
  // put your main code here, to run repeatedly:
  // for(;;);
  sprintf(msg, "Total heap: %u", ESP.getHeapSize());
  Serial.println(msg);
  sprintf(msg, "Free heap: %u\n", ESP.getFreeHeap());
  Serial.println(msg);
  sprintf(msg, "Total PSRAM: %u", ESP.getPsramSize());
  Serial.println(msg);
  sprintf(msg, "Free PSRAM: %d\n", ESP.getFreePsram());
  Serial.println(msg);

  // log_d("Total heap: %d", ESP.getHeapSize());
  // log_d("Free heap: %d", ESP.getFreeHeap());
  // log_d("Total PSRAM: %d", ESP.getPsramSize());
  // log_d("Free PSRAM: %d", ESP.getFreePsram());

  // sprintf(msg, "spiram size %u", esp_spiram_get_size());
  // Serial.println(msg);
  // sprintf(msg, "himem free %u", esp_himem_get_free_size());
  // Serial.println(msg);
  // sprintf(msg, "himem phys %u", esp_himem_get_phys_size());
  // Serial.println(msg);
  // sprintf(msg, "himem reserved %u", esp_himem_reserved_area_size());
  delay(1000);
}