#include <Arduino.h>
#include "BluetoothSerial.h"

#define y 0
#define y_ 1
#define b 10
#define b_ 11
#define x_ 20
#define ack 999

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32test"); //Bluetooth device name
  Serial.println("The device started, now you can pair it with bluetooth!");
}

void loop() {
  if(SerialBT.hasClient() == 0)
  {
    Serial.println("No client connected");
  }
  else
  {
    if (Serial.available()) {
      SerialBT.write(Serial.read());
    }
    if (SerialBT.available()) {
      Serial.write(SerialBT.read());
    }
    SerialBT.println("Testing");
  }
  
  delay(100);
}
