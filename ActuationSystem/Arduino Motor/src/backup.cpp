// #include <Arduino.h>

// // defines
// #define STEP_PIN  33
// #define DIR_PIN   32
// #define EN_PIN    14
// #define DELAY     700

// void setup() {
//   // setup pins
//   pinMode(STEP_PIN, OUTPUT);
//   pinMode(DIR_PIN, OUTPUT);
//   pinMode(EN_PIN, OUTPUT);

//   digitalWrite(EN_PIN, LOW);
// }

// void loop() {
//   digitalWrite(DIR_PIN, HIGH);

//   for(int x = 0; x < 800; x++)
//   {
//     digitalWrite(STEP_PIN, HIGH);
//     delayMicroseconds(DELAY);
//     digitalWrite(STEP_PIN, LOW);
//     delayMicroseconds(DELAY);
//   }

//   delay(1000);

//   digitalWrite(DIR_PIN, LOW);

//   for(int x = 0; x < 800; x++)
//   {
//     digitalWrite(STEP_PIN, HIGH);
//     delayMicroseconds(DELAY);
//     digitalWrite(STEP_PIN, LOW);
//     delayMicroseconds(DELAY);
//   }

//   delay(1000);
// }