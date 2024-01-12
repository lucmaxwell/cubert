#include <Arduino.h>
#include <SerialCommands.h>
#include "BluetoothSerial.h"

//pin assignments
#define motors_en_pin               5         // LOW: Driver enabled. HIGH: Driver disabled
#define motors_base_step_pin        2  // Step on rising edge
#define motors_base_dir_pin        15
#define motors_arm_left_dir_pin     0
#define motors_arm_left_step_pin    4
#define motors_arm_right_dir_pin   16
#define motors_arm_right_step_pin  17
#define endstop_arm_openLimit_pin  18
#define endstop_arm_upperLimit_pin 23
#define endstop_arm_lowerLimit_pin 19

// code clarification definitions
#define cw        0
#define ccw       1
#define UP        1
#define DOWN      0
#define OPEN      1
#define CLOSE     0
#define MIDDLE  420
#define BOTTOM   endstop_arm_lowerLimit_pin
#define TOP      endstop_arm_upperLimit_pin

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bluetooth/cubing notation
#define y 'y'
#define yp 'Y'
#define b 'b'
#define bp 'B'
#define xp 'X'
#define ACK 'a'
#define OK 'k'
#define END '\r'

BluetoothSerial SerialBT;
int8_t BluetoothIn;

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled, run `make menuconfig` to and enable it
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_SPEED 3         // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
#define MIN_SPEED 0.000001  // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
///////////////////////////////////////////////////////////////////////////////////////////////////////////
int numStepsToGripOrUngrip     =     100;
int gripStrength               =     375;
int moveArmSpeed               =      10;        // set the velocity (1-100) that we will raise or lower the arm
int handOpenCloseSpeed         =      15;  // set the velocity (1-100) that we will open and close the ha
int spinSpeed                  =     100;
int interActionDelay           =      10;
int cubeDropDistance           =     400;
int numStepsFromBottomToMiddle =     800;
int numStepsFromTopToMiddle    =    1100;
float numDegreesToRotate       =      93;
int homePosition               =  MIDDLE;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
String handState   = "unknown state";
String armLocation = "unknown state";

//manual button pin assignments
#define raiseArmButton  33
#define lowerArmButton  26
#define openHandButton  25
#define closeHandButton 32
#define spinBaseButton  27

static bool gripperFunctional = true;

void moveArm(int direction);
void homeArmAndHand();
void moveArmTo(int destination);
void closeHand();
void openHand();
int getDelay(int v);
int getIntegerFromUser();
void articulateHand(int direction);

char serial_command_buffer_[32];
SerialCommands serial_commands_(&Serial, serial_command_buffer_, sizeof(serial_command_buffer_), "\r\n", " ");
void cmd_unrecognized(SerialCommands* sender, const char* cmd) {
sender->GetSerial()->println("Unrecognized command");
}
void homeArmAndHand() {
  String desiredLocation = "";
  Serial.print("Homing hand to ");
  switch (homePosition) {
    case TOP:
      desiredLocation = "top";
      break;
    case BOTTOM:
      desiredLocation = "bottom";
      break;
    case MIDDLE:
      desiredLocation = "middle";
      break;
  }
  Serial.println(desiredLocation);

  openHand();
  if(gripperFunctional){
  moveArmTo(homePosition);
  }
}
void moveArmTo(int destination) {
  String desiredLocation = "";
  Serial.print("Moving arm from ");
  Serial.print(armLocation);
  Serial.print(" to ");
  switch (destination) {
    case TOP:
      desiredLocation = "top";
      break;
    case BOTTOM:
      desiredLocation = "bottom";
      break;
    case MIDDLE:
    desiredLocation = "middle";
    break;
  }
  Serial.println(desiredLocation);
  switch (destination) {
    /////////////////////////////////////////////////////////
    case BOTTOM:
      while (1) {
        if (digitalRead(destination) == 0)
          break;
        moveArm(DOWN);
      }
      armLocation = "bottom";
      break;
    /////////////////////////////////////////////////////////
    case TOP:
      while(1){
        if(digitalRead(destination) == 0)
          break;
      moveArm(UP);
      }     
      armLocation = "top";
      break;
    ///MIDDLE/////////////////////////////////////////////////////
    case MIDDLE:
      if(armLocation.equals("middle")){
        break;
      }
      if(armLocation.equals("top")){          ;
          for(int i = 0; i < numStepsFromTopToMiddle; i++){
            moveArm(DOWN);
            break;
          }
      }
      else if (armLocation.equals("bottom")){
          Serial.println(" from the bottom.");
          for(int i = 0; i < numStepsFromBottomToMiddle; i++){
            moveArm(UP);
            break;
          }
      } 
      else {
        while (digitalRead(BOTTOM) == 1) {     
          moveArm(DOWN);
        }
        delay(interActionDelay);
        for(int i = 0; i < numStepsFromBottomToMiddle; i++){
              moveArm(UP);            
        }
      }
      armLocation = "middle";
    break;    
  }
    Serial.println("Arrived at destination.");
    delay(interActionDelay);
}
void closeHand() {
  for (int i = 0; i < gripStrength; i++) {
    articulateHand(CLOSE);
  }
  delay(interActionDelay);
  handState = "closed"; 
}  
void openHand() {

  Serial.println("Opening hand");
  int count = 0;
  while (1) {
    if (digitalRead(endstop_arm_openLimit_pin) == 0)
      return;
    articulateHand(OPEN);
    count++;
    if (count > 50000){
        gripperFunctional = false;
        break;
    }
  }
  handState = "open";
  delay(interActionDelay); 
}
void spinBase(int my_direction, float deg, int v) {
  int stepDelay = getDelay(v);
  if (armLocation != "top" && armLocation != "middle" && gripperFunctional) {
    moveArmTo(MIDDLE);
  }

  digitalWrite(motors_base_dir_pin, my_direction);  // set the direction
  float steps = 19200 * deg / 360.0;
  for (int i = 0; i < steps; i++) {
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
    delayMicroseconds(stepDelay);
  }
  // correct for over/under rotation of the cube
  digitalWrite(motors_base_dir_pin, !my_direction);  // set the direction
  openHand();
  float correction = deg - 90;
  steps = 19200 * correction / 360.0;
  for (int i = 0; i < steps; i++) {
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
    delayMicroseconds(stepDelay);
  }

}
void moveArm(int direction) {
  armLocation = "undefined";
  //set the direction that the arm motors spin
  if (direction == UP) {
    // Serial.println("Going up?");
    digitalWrite(motors_arm_left_dir_pin, cw);
    digitalWrite(motors_arm_right_dir_pin, ccw);
  } else {
    // Serial.println("Going down?");
    digitalWrite(motors_arm_left_dir_pin, ccw);
    digitalWrite(motors_arm_right_dir_pin, cw);
  }
  if (direction == DOWN && digitalRead(endstop_arm_lowerLimit_pin) == 0) {
    // Serial.println("Can't go any lower");
    return;
  }
  if (direction == UP && digitalRead(endstop_arm_upperLimit_pin) == 0) {
    // Serial.println("Can't go any higher");
    return;
  }
  digitalWrite(motors_arm_left_step_pin, !digitalRead(motors_arm_left_step_pin));    // Step left motor
  digitalWrite(motors_arm_right_step_pin, !digitalRead(motors_arm_right_step_pin));  // Step right motor
  // delay(5);
  int stepDelay = getDelay(moveArmSpeed);
  delayMicroseconds(stepDelay);
}
void dropCubeToBase() {
  for (int i = 0; i < cubeDropDistance; i++) {
    moveArm(DOWN);
  }
  delay(interActionDelay);
  armLocation = "unknown state"; // lol imagine forgetting to put this here xD
}
void articulateHand(int direction) {
  handState = "undefined";
  //set the direction that the arm motors spin
  if (direction == OPEN) {
    // Serial.println("Opening");
    digitalWrite(motors_arm_left_dir_pin, cw);
    digitalWrite(motors_arm_right_dir_pin, cw);
  } else {
    // Serial.println("Closing");
    digitalWrite(motors_arm_left_dir_pin, ccw);
    digitalWrite(motors_arm_right_dir_pin, ccw);
  }

  if (digitalRead(endstop_arm_openLimit_pin) == 1 || direction == CLOSE) {
    digitalWrite(motors_arm_left_step_pin, !digitalRead(motors_arm_left_step_pin));    // Step left motor
    digitalWrite(motors_arm_right_step_pin, !digitalRead(motors_arm_right_step_pin));  // Step right motor
  }
  int stepDelay = getDelay(handOpenCloseSpeed);
  delayMicroseconds(stepDelay);
}
void clockwise(SerialCommands * sender) {
  spinBase(cw, numDegreesToRotate, spinSpeed);
}
void counterClockwise(SerialCommands * sender) {
  spinBase(ccw, numDegreesToRotate, spinSpeed);
  sender->GetSerial()->println("Turning counter clockwise.");
}
int getIntegerFromUser() {
  Serial.println("Please enter a value:");
  while (!Serial.available()) {}  // wait until the user enters a value
  int value = Serial.parseInt();
  while (Serial.available()) {  // flush the serial input buffer
    int flush = Serial.read();
  }
  return value;
}
void displayMe(SerialCommands * sender) {
  while (!Serial.available()) {
    spinBase(cw, 360, 1);
    sender->GetSerial()->println("Keep rollin rollin rollin rollin...");
  }
}
void fast(SerialCommands * sender) {
  while (!Serial.available()) {
    spinBase(cw, 360, 100);
    sender->GetSerial()->println("Keep rollin rollin rollin rollin...");
  }
}
void flipCube() {
  openHand();
  moveArmTo(BOTTOM);
  closeHand();
  moveArmTo(TOP);
  dropCubeToBase();
  openHand();
}
void rotateFace(int dir, int deg, int v) {
  moveArmTo(MIDDLE);
  closeHand();
  spinBase(dir, deg, v);
  openHand();
  armLocation = "middle";
  handState = "open";
  delay(interActionDelay);
}
void scramble(SerialCommands * sender) {
  int numScrambles = 0;
  if(armLocation.equals("unknown state"))
    homeArmAndHand();
  while(numScrambles < 13) {
    int choice = random(0, 3);
    int randomDirection = random(0, 2);
    switch (choice) {
      case 0:
        Serial.println("//////////////////// Flipping cube");
        flipCube();
        break;
      case 1:
        Serial.println("//////////////////// Spinning base");
        spinBase(randomDirection, numDegreesToRotate, spinSpeed);
        Serial.println("spinning base");
        break;
      case 2:
        Serial.println("//////////////////// rotating face of cube");
        rotateFace(randomDirection, numDegreesToRotate, spinSpeed);
        Serial.println("rotating base");
        numScrambles++;
        break;
    }
  }
}
void zenMode(SerialCommands * sender){

  int defaultMoveArmSpeed        = moveArmSpeed;         // get the current speed settings
  int defaultHandOpenCloseSpeed  = handOpenCloseSpeed;;  
  int defaultSpinSpeed           = spinSpeed;

  if(armLocation.equals("unknown state") || handState.equals("unknown state") ){
    homeArmAndHand();
  }

  moveArmSpeed       =  5; //change these to make Cubert whisper quiet
  handOpenCloseSpeed   =  1;
  spinSpeed            = 20;

  while(!Serial.available()) { // scramble until Yoshida stops being a poor lecturer (aka indefinitely)
    int choice = random(0, 3);
    int randomDirection = random(0, 2);
    switch (choice) {
      case 0:
        Serial.println("//////////////////// Flipping cube");
        flipCube();
        break;
      case 1:
        Serial.println("//////////////////// Spinning base");
        spinBase(randomDirection, numDegreesToRotate, spinSpeed);
        Serial.println("spinning base");
        break;
      case 2:
        Serial.println("//////////////////// rotating face of cube");
        rotateFace(randomDirection, numDegreesToRotate, spinSpeed);
        Serial.println("rotating base");       
        break;
    }
  }
  moveArmSpeed       = defaultMoveArmSpeed;        // set it back
  handOpenCloseSpeed = defaultHandOpenCloseSpeed;  // set it back
  spinSpeed          = defaultSpinSpeed;

}
void testDistanceToMiddleFromTop(SerialCommands * sender) {
  int value = getIntegerFromUser();
  moveArmTo(TOP);
  for (int i = 0; i < value; i++) {
    moveArm(DOWN);
  }
}
void testDistanceToMiddleFromBottom(SerialCommands * sender) {
  int value = getIntegerFromUser();
  moveArmTo(BOTTOM);
  for (int i = 0; i < value; i++) {
    moveArm(UP);
  }
}
void rotateFace(int direction, float degrees, int v) {
  moveArmTo(MIDDLE);
  closeHand();
  spinBase(direction, degrees, v);
  openHand();
  armLocation = "middle";
  handState = "open";
}
void testGripDistance(SerialCommands * sender) {
  while (digitalRead(endstop_arm_openLimit_pin) == 1) {
    articulateHand(OPEN);
  }
  Serial.println("Please enter how many steps to close the gripper for:");
  int numSteps = getIntegerFromUser();
  Serial.print("Now closing gripper for ");
  Serial.print(numSteps, DEC);
  Serial.println(" steps.");
  for (int i = 0; i < numSteps; i++) {
    articulateHand(CLOSE);
  }
}
void speeen(SerialCommands * sender) {
  int startTime = 0;
  openHand();
  while (!Serial.available()) {
    startTime = millis();
    moveArmTo(BOTTOM);
    closeHand();
    moveArmTo(TOP);
    dropCubeToBase();
    openHand();

    int elapsed = millis() - startTime;
    Serial.print("Time to spin = ");
    Serial.print(elapsed, DEC);
    Serial.println(" ms.");
  }
}

//initialize serialCommand functions
//
SerialCommand clockwise_("r", clockwise);
SerialCommand scramble_("scramble", scramble);
SerialCommand counterClockwise_("l", counterClockwise);
SerialCommand displayMe_("d", displayMe);
SerialCommand fast_("f", fast);
SerialCommand testGripDistance_("t", testGripDistance);
SerialCommand speeen_("s", speeen);
SerialCommand testDistanceToMiddleFromTop_("m", testDistanceToMiddleFromTop);
SerialCommand testDistanceToMiddleFromBottom_("m2", testDistanceToMiddleFromBottom);
SerialCommand zenMode_("zen", zenMode);

void setup() {
  Serial.begin(115200);
  serial_commands_.SetDefaultHandler(cmd_unrecognized);
  serial_commands_.AddCommand(&clockwise_);
  serial_commands_.AddCommand(&counterClockwise_);
  serial_commands_.AddCommand(&displayMe_);
  serial_commands_.AddCommand(&fast_);
  serial_commands_.AddCommand(&testGripDistance_);
  serial_commands_.AddCommand(&speeen_);
  serial_commands_.AddCommand(&testDistanceToMiddleFromTop_);
  serial_commands_.AddCommand(&testDistanceToMiddleFromBottom_);
  serial_commands_.AddCommand(&scramble_);
  serial_commands_.AddCommand(&zenMode_);

  // configure motor pins for esp32 wroom
  pinMode(motors_en_pin, OUTPUT);
  pinMode(motors_base_step_pin, OUTPUT);
  pinMode(motors_base_dir_pin, OUTPUT);
  pinMode(motors_arm_left_step_pin, OUTPUT);
  pinMode(motors_arm_left_dir_pin, OUTPUT);
  pinMode(motors_arm_right_step_pin, OUTPUT);
  pinMode(motors_arm_right_dir_pin, OUTPUT);

  //configure control buttons
  pinMode(raiseArmButton, INPUT_PULLUP);
  pinMode(lowerArmButton, INPUT_PULLUP);
  pinMode(openHandButton, INPUT_PULLUP);
  pinMode(closeHandButton, INPUT_PULLUP);
  pinMode(spinBaseButton, INPUT_PULLUP);
  pinMode(endstop_arm_openLimit_pin, INPUT_PULLUP);
  pinMode(endstop_arm_upperLimit_pin, INPUT_PULLUP);
  pinMode(endstop_arm_lowerLimit_pin, INPUT_PULLUP);
  digitalWrite(motors_en_pin, LOW);

  // Start bluetooth
  SerialBT.begin("ESP32test"); //Bluetooth device name

  Serial.println("Ready!");
  homeArmAndHand();
}

void loop() {
  int raise = digitalRead(raiseArmButton);
  int lower = digitalRead(lowerArmButton);
  int open  = digitalRead(openHandButton);
  int close = digitalRead(closeHandButton);
  int spin  = digitalRead(spinBaseButton);

  if (raise == 0) {
    moveArm(UP);
  }

  if (lower == 0) {
    moveArm(DOWN);
  }

  if (open == 0) {
    articulateHand(OPEN);
  }

  if (close == 0) {
    articulateHand(CLOSE);
  }

  if (spin == 0) {
    spinBase(cw, numDegreesToRotate, spinSpeed);
  }

  serial_commands_.ReadSerial();

  // Handle bluetooth connection
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
      BluetoothIn = SerialBT.read();
      Serial.write("Received: ");
      Serial.write(BluetoothIn);
      Serial.println("");
      
      // Do command 
      switch (BluetoothIn)
      {
        case y:
          clockwise(NULL);
          break;
          
        case yp:
          counterClockwise(NULL);
          break;

        case b:
          clockwise(NULL);
          break;

        case bp:
          counterClockwise(NULL);
          break;

        case xp:
          speeen(NULL);
          SerialBT.write(xp);
          break;
      }

      SerialBT.write(OK);
    }
  }
  delay(10);
}

int getDelay(int v) {
  v = min(v, 120);
  double x = MIN_SPEED + v * (MAX_SPEED - MIN_SPEED) / 100;
  double delayDuration = pow(0.0003 * x, -1) / 10;
  return round(delayDuration);
}