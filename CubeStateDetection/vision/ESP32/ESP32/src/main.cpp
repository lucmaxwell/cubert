#include <Arduino.h>
#include "BluetoothSerial.h"

#define y 'y'
#define yp 'Y'
#define b 'b'
#define bp 'B'
#define xp 'X'
#define ack 'a'

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

BluetoothSerial SerialBT;
int8_t in;

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
      in = SerialBT.read();
      Serial.write("Received: ");
      Serial.write(in);
      Serial.println("");

      if(in == *"\r")
      {
        Serial.println("Received end of message signal");
        SerialBT.write(ack);
      }
    }
  }
  delay(100);
}