#include <Arduino.h>
#include <SerialCommands.h>
#include "BluetoothSerial.h"

//motor/endstop/current sense pin assignments
#define motors_en_pin               5      // LOW: Driver enabled. HIGH: Driver disabled
#define motors_base_step_pin        2      // Step on rising edge
#define motors_base_dir_pin        15
#define motors_arm_left_dir_pin     0
#define motors_arm_left_step_pin    4
#define motors_arm_right_dir_pin   16
#define motors_arm_right_step_pin  17
#define endstop_arm_openLimit_pin  18
#define endstop_arm_upperLimit_pin 23
#define endstop_arm_lowerLimit_pin 19
#define currentSensor_pin          34

//manual button pin assignments
#define raiseArmButton  33
#define lowerArmButton  26
#define openHandButton  25
#define closeHandButton 32
#define spinBaseButton  27
//

// code clarification definitions
#define UNKNOWN         -1
#define cw               0
#define ccw              1
#define UP               1
#define DOWN             0
#define OPEN             1
#define CLOSE            0
#define CLOSED           0
#define OPENED           0
#define MIDDLE         420
#define DROPOFFHEIGHT   69
#define BOTTOM          19
#define TOP             23
#define FRONTCW          1
#define FRONTCCW         2
#define REARCW           3
#define REARCCW          4
#define TOPCW            5
#define TOPCCW           6
#define LEFTCW           7
#define LEFTCCW          8
#define RIGHTCW          9
#define RIGHTCCW        10
#define BOTTOMCW        11
#define BOTTOMCCW       12
//////////////////////// ///////////////////////////////////////////////////////////////////////////////////
// Bluetooth/cubing notation
#define y    'y'
#define yp   'Y'
#define b    'b'
#define bp   'B'
#define xp   'X'
#define ACK  'a'
#define OK   'k'
#define END '\r'

BluetoothSerial SerialBT;
int8_t BluetoothIn;

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled, run `make menuconfig` to and enable it
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_SPEED 3.3        // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
#define MIN_SPEED 0.000001  // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
///////////////////////////////////////////////////////////////////////////////////////////////////////////
int numStepsToGripOrUngrip      =     100;
int gripStrength                =     450;
int moveArmSpeed                =      50;        // set the velocity (1-100) that we will raise or lower the arm
int handOpenCloseSpeed          =      50;  // set the velocity (1-100) that we will open and close the ha
int spinSpeed                   =     120;
int interActionDelay            =      10;
int cubeDropDistance            =     400;
int numStepsFromBottomToMiddle  =     800;
int numStepsFromTopToMiddle     =    1100;
int numStepsFromDropoffToMiddle =     700;
float cubeRotationError         =       4; // FLAG - This is currently set for Bruno's cube. Whatever this number is for other cubes needs to be calculated using comp. vision
int homePosition                =  MIDDLE;
int zenSpinSpeed                =      10;
int zenArmSpeed                 =      10;
int zenHandOpenCloseSpeed       =      10;
int baseGearPlayCompensation    =      20; // I think that when we correct, the first few steps are used to
                                           // start switching directions (ie, there is some play in the spin
                                           //gear, causing error to build up every time we correct)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
int handState    = UNKNOWN;
int armLocation  = UNKNOWN;
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
sender->GetSerial()->println("Unrecognized command");}
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
  }}
void moveArmTo(int destination) {
  if(!gripperFunctional) 
  {
    return;
  }  
  String desiredLocation = "";
  Serial.print("Moving arm from ");
  switch(armLocation){
    case TOP:
      Serial.print("top");
      break;
    case MIDDLE:
      Serial.print("middle");
      break;
    case BOTTOM:
      Serial.print("bottom");
      break;
    case UNKNOWN:
      Serial.print("unknown location");
      break;
  }
  
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
      armLocation = BOTTOM;
      break;
    /////////////////////////////////////////////////////////
    case TOP:
      while(1){
        if(digitalRead(destination) == 0)
          break;
      moveArm(UP);
      }     
      armLocation = TOP;
      break;
    ///MIDDLE/////////////////////////////////////////////////////
    case MIDDLE:
      if(armLocation == MIDDLE){
        break;
      }
      if(armLocation == TOP){                    
          for(int i = 0; i < numStepsFromTopToMiddle; i++){
            moveArm(DOWN);
          }
          armLocation = MIDDLE;
          break;
      }
      else if (armLocation == BOTTOM ){
          Serial.println(" from the bottom.");
          for(int i = 0; i < numStepsFromBottomToMiddle; i++){
            moveArm(UP);          
          }
          armLocation = MIDDLE;
          break;
      } 
      else if (armLocation == DROPOFFHEIGHT){
        for(int i = 0; i < numStepsFromDropoffToMiddle; i++){
          moveArm(DOWN);
        }
        armLocation = MIDDLE;
        break;
      }
      else {
        while (digitalRead(TOP) == 1) {     
          moveArm(UP);
        }
        delay(interActionDelay);
        for(int i = 0; i < numStepsFromTopToMiddle; i++){
              moveArm(DOWN);            
        }
      }
      armLocation = MIDDLE;
      break;    
      
  }
    Serial.println("Arrived at destination.");
    delay(interActionDelay);}
void closeHand() {
  Serial.println("Closing hand");
  for (int i = 0; i < gripStrength; i++) {
    articulateHand(CLOSE);
  }
  delay(interActionDelay);
  handState = CLOSE; }  
void openHand() {

  Serial.println("Opening hand");
  int count = 0;
  while (1) {
    if (digitalRead(endstop_arm_openLimit_pin) == 0)
      return;
    articulateHand(OPEN);
    count++;
    if (count > 10000){
        gripperFunctional = false;
        Serial.println("Hand could not reach end stop");
        break;
    }
  }
  handState = OPENED;
  delay(interActionDelay); }
void spinBase(int my_direction, bool correctionEnabled) {
  int stepDelay = getDelay(spinSpeed);
  int degreesToRotate = 90;   

  if(correctionEnabled){
    degreesToRotate += cubeRotationError;
  } 
  
  if (armLocation != TOP && armLocation != MIDDLE && gripperFunctional) {
    openHand();
    moveArmTo(MIDDLE);
  }

  digitalWrite(motors_base_dir_pin, my_direction);  // set the direction
  int steps = 19200 * degreesToRotate / 360;
  for (int i = 0; i < steps; i++) {
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
    delayMicroseconds(stepDelay);
  }
  delay(interActionDelay);
  
  if(correctionEnabled){
    if(my_direction == cw){
      my_direction = ccw;
    }else{
      my_direction = cw;
    }  
    // correct for over/under rotation of the cube
    digitalWrite(motors_base_dir_pin, my_direction);  // set the direction
    openHand();  

    int steps = 19200 * cubeRotationError / 360.0;
    for (int i = 0; i < steps; i++) {
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
      delayMicroseconds(stepDelay);
    }
    delay(interActionDelay);
  }  
  }
void spinBaseTwice(int my_direction, bool correctionEnabled){
  int defaultInteractionDelay = interActionDelay;
  interActionDelay = 0;
  spinBase(my_direction, false);
  interActionDelay = defaultInteractionDelay;
  spinBase(my_direction, true);  
  }

void moveArm(int direction) {
  armLocation = UNKNOWN;
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
  delayMicroseconds(stepDelay);}
void dropCubeToBase() {
  for (int i = 0; i < cubeDropDistance; i++) {
    moveArm(DOWN);
  }
  delay(interActionDelay);
  armLocation = UNKNOWN; // lol imagine forgetting to put this here xD
  armLocation = DROPOFFHEIGHT;}
void articulateHand(int direction) {
  handState = UNKNOWN;
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
  delayMicroseconds(stepDelay);}

int  getDelay(int v) {
  v = min(v, 200);
  double x = MIN_SPEED + v * (MAX_SPEED - MIN_SPEED) / 100;
  double delayDuration = pow(0.0003 * x, -1) / 10;
  return round(delayDuration);}

int  getIntegerFromUser() {
  Serial.println("Please enter a value:");
  while (!Serial.available()) {}  // wait until the user enters a value
  int value = Serial.parseInt();
  while (Serial.available()) {  // flush the serial input buffer
    int flush = Serial.read();
  }
  return value;}

void flipCube() {
  Serial.println("////// Flipping cube");
  openHand();
  moveArmTo(BOTTOM);
  closeHand();
  moveArmTo(TOP);
  delay(50);
  dropCubeToBase();
  openHand();}

void rotateFace(int face, int singleOrDouble) {
  int direction;   
  openHand();
  switch(face){
    case FRONTCW:
      Serial.println("/////////////// Turning front face clockwise");
      direction = cw;
      flipCube();
      break;
    case FRONTCCW:
      Serial.println("/////////////// Turning front face counterclockwise");
      direction = ccw;
      flipCube();
      break;
    case REARCW:
      Serial.println("/////////////// Turning rear face clockwise");
      direction = ccw;
      spinBaseTwice(ccw, false);        
      flipCube();
      break;
    case REARCCW:
      Serial.println("/////////////// Turning rear face counterclockwise");
      direction = cw;
      spinBaseTwice(cw, false);        
      flipCube();
      break;
    case RIGHTCW:
      Serial.println("/////////////// Turning right face clockwise");
      direction = ccw;
      spinBase(ccw, false);        
      flipCube();
      break;
    case RIGHTCCW:
      Serial.println("/////////////// Turning right face counterclockwise");
      direction = ccw;
      spinBase(cw, false);        
      flipCube();
      break;
    case LEFTCW:
      Serial.println("/////////////// Turning left face clockwise");
      direction = cw;
      spinBase(cw, false);        
      flipCube();
      break;
    case LEFTCCW:
      Serial.println("/////////////// Turning left face counterclockwise");
      direction = ccw;
      spinBase(cw, false);        
      flipCube();
      break;
    case TOPCW:
      Serial.println("/////////////// Turning top face clockwise");
      direction = cw;
      flipCube();
      flipCube();
      break;
    case TOPCCW:
        Serial.println("/////////////// Turning top face counterclockwise");
        direction = ccw;
        flipCube();
        flipCube();
        break;
    case BOTTOMCW:
        Serial.println("/////////////// Turning bottom face clockwise");
        direction = cw;
        break;
    case BOTTOMCCW:
        Serial.println("/////////////// Turning bottom face counterclockwise");
        direction = ccw;
        break;
  }
  moveArmTo(MIDDLE); 
  closeHand();
  if(singleOrDouble == 2){
    spinBaseTwice(direction, true);
  }else{
    spinBase(direction, true);
  }

  openHand();
  armLocation = MIDDLE;
  handState   = OPENED;
  delay(interActionDelay);}





void testCorrection(SerialCommands *sender){
  moveArmTo(MIDDLE);
  closeHand();
  Serial.println("How many degrees to correct?");
  cubeRotationError = getIntegerFromUser();
  spinBase(cw, true);
 }
void testDistanceToMiddleFromTop(SerialCommands * sender) {
  int value = getIntegerFromUser();
  moveArmTo(TOP);
  for (int i = 0; i < value; i++) {
    moveArm(DOWN);
  }}
void testDistanceToMiddleFromBottom(SerialCommands * sender) {
  int value = getIntegerFromUser();
  moveArmTo(BOTTOM);
  for (int i = 0; i < value; i++) {
    moveArm(UP);
  }}
void setDistanceToMiddleFromCubeRelease(SerialCommands *sender){
  openHand();
  moveArmTo(TOP);
  for(int i = 0; i < cubeDropDistance; i++){
    moveArm(DOWN);
  }
  Serial.println("Please enter how many steps to travel from the dropoff location to the middle of the cube: ");
  int numSteps = getIntegerFromUser();

  for(int i = 0; i < numSteps; i++){
    moveArm(DOWN);
  }
  Serial.println("Set distance from dropoff to middle");
  numStepsFromDropoffToMiddle = numSteps; }
void setInterActionDelay(SerialCommands *sender){
  Serial.println("Please enter an inter-action delay (ms)");
  interActionDelay = getIntegerFromUser();}
void setScrambleAndSolveSpeed(SerialCommands *sender){
  Serial.println("Please enter the cube spin speed (0 - 120):");
  spinSpeed  = getIntegerFromUser();
  Serial.print("Set cube spin speed to "); Serial.println(spinSpeed);

  Serial.println("Please enter the arm move speed (0 - 100):");
  moveArmSpeed   = getIntegerFromUser();  
  Serial.print("Set arm speed to "); Serial.println(moveArmSpeed);

  Serial.println("Please enter the hand open/close speed (0 - 100):");
  handOpenCloseSpeed   = getIntegerFromUser();  
  Serial.print("Set hand speed to "); Serial.println(handOpenCloseSpeed);}

void setCubeError(SerialCommands *sender){
  Serial.println("Please set the cube rotation error:");
  int value = getIntegerFromUser();
  cubeRotationError = value;}

void setGripDistance(SerialCommands * sender) {
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
  gripStrength = numSteps;
  Serial.print("Set grip strength to ");
  Serial.println(numSteps);
  armLocation = UNKNOWN;
  handState = UNKNOWN;}

void setZenSpeeds(SerialCommands * sender){
  Serial.println("Please enter the cube spin speed for zen mode (0 - 100):");
  zenSpinSpeed  = getIntegerFromUser();
  Serial.print("Set cube spin speed to "); Serial.println(zenSpinSpeed);

  Serial.println("Please enter the arm move speed for zen mode (0 - 100):");
  zenArmSpeed   = getIntegerFromUser();  
  Serial.print("Set arm speed to "); Serial.println(zenArmSpeed);

  Serial.println("Please enter the hand open/close speed for zen mode (0 - 100):");
  zenHandOpenCloseSpeed   = getIntegerFromUser();  
  Serial.print("Set arm speed to "); Serial.println(zenArmSpeed);}
void toggleSteppers(SerialCommands *sender){
  digitalWrite(motors_en_pin,!digitalRead(motors_en_pin));
  Serial.println("Switched motors state (on/off)");
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
  }}
void testMove(SerialCommands *sender){
  Serial.println("\n\n\n\n\n\n\n\n\n");
  Serial.println("Which move would you like to try? (1-10)");
  Serial.println(" 1) Rotate front face clockwise");
  Serial.println(" 2) Rotate front face counterclockwise");
  Serial.println(" 3) Rotate rear face clockwise");
  Serial.println(" 4) Rotate rear face counterclockwise");
  Serial.println(" 5) Rotate top face clockwise");
  Serial.println(" 6) Rotate top face counterclockwise");
  Serial.println(" 7) Rotate left face clockwise");
  Serial.println(" 8) Rotate left face counterclockwise");
  Serial.println(" 9) Rotate right face clockwise");
  Serial.println("10) Rotate right face counterclockwise");
  Serial.println("11) Rotate bottom face clockwise");
  Serial.println("12) Rotate bottom face counterclockwise");

  int moveNumber = getIntegerFromUser();
  rotateFace(moveNumber, 1);}



void zenMode(SerialCommands *sender){
  int defaultSpinSpeed          = spinSpeed;
  int defaultMoveArmSpeed       = moveArmSpeed;        // set the velocity (1-100) that we will raise or lower the arm
  int defaultHandOpenCloseSpeed = handOpenCloseSpeed;
  // change to nice slow speeds for zen mode
  spinSpeed          = zenSpinSpeed;
  moveArmSpeed       = zenArmSpeed;
  handOpenCloseSpeed = zenHandOpenCloseSpeed;
  int lastMove = TOPCW;
  while(!Serial.available()){
    int randomNumberOfTurns = random(1,2); //FLAG  
    int randomMove = random(1,11);
    rotateFace(randomMove, randomNumberOfTurns);  
    lastMove = randomMove; //FLAG - this does not actualyl preven the previous move to be cancelled out due to the transformation that occurs to the cube when it is flipped ie topCW for move2
    //isn't opped by topCCW on move 1
  }

  //revert to hardcoded default speeds
  spinSpeed          = defaultSpinSpeed;
  moveArmSpeed       = defaultMoveArmSpeed;
  handOpenCloseSpeed = defaultHandOpenCloseSpeed;
  }

void scramble(SerialCommands * sender) {
  int lastMove = TOPCW;  
  int numScrambles = 13;
  float startTime = millis();
  for(int i = 0; i < numScrambles; i++){ 
    int randomNumberOfTurns = random(1,3);    
    int randomMove = random(1,13);
    rotateFace(randomMove, randomNumberOfTurns); 
    Serial.print("Performing ");
    Serial.print(randomNumberOfTurns);
    Serial.println(" turns.");
    lastMove = randomMove;
  }
  float elapsed = millis() - startTime;
  Serial.print("It took "); Serial.print(elapsed/1000);Serial.print(" seconds to perform "); Serial.print(numScrambles);Serial.println(" scrambles."); 
}//initialize serialCommand functions

SerialCommand setGripDistance_("t", setGripDistance);
SerialCommand testDistanceToMiddleFromTop_("m", testDistanceToMiddleFromTop);
SerialCommand testDistanceToMiddleFromBottom_("m2", testDistanceToMiddleFromBottom);
SerialCommand setCubeError_("configCube", setCubeError);
SerialCommand setDistanceToMiddleFromCubeRelease_("d2", setDistanceToMiddleFromCubeRelease);
SerialCommand setZenSpeeds_("setZen", setZenSpeeds);
SerialCommand testMove_("testMove", testMove);
SerialCommand testCorrection_("testCorr", testCorrection);
SerialCommand toggleSteppers_("tog", toggleSteppers);
SerialCommand setInterActionDelay_("setD", setInterActionDelay);
SerialCommand setScrambleAndSolveSpeed_("setSpeed", setScrambleAndSolveSpeed);

SerialCommand speeen_("speeen", speeen);
SerialCommand zenMode_("zen", zenMode); // zen mode will slowly and infinitely scramble until serial input is received (will finish current move)
SerialCommand scramble_("scramble", scramble); // scramble will ask the user for a number of moves to scramble for and then perform that many random scrambles


void setup() {
  Serial.begin(115200);
  serial_commands_.SetDefaultHandler(cmd_unrecognized);

  serial_commands_.AddCommand(&setGripDistance_);
  serial_commands_.AddCommand(&testDistanceToMiddleFromTop_);
  serial_commands_.AddCommand(&testDistanceToMiddleFromBottom_);
  serial_commands_.AddCommand(&setDistanceToMiddleFromCubeRelease_);  
  serial_commands_.AddCommand(&setCubeError_);
  serial_commands_.AddCommand(&testMove_); 
  serial_commands_.AddCommand(&setZenSpeeds_); 
  serial_commands_.AddCommand(&toggleSteppers_); 
  serial_commands_.AddCommand(&setInterActionDelay_);
  serial_commands_.AddCommand(&setScrambleAndSolveSpeed_); 

  serial_commands_.AddCommand(&speeen_);
  serial_commands_.AddCommand(&scramble_);
  serial_commands_.AddCommand(&zenMode_);
  serial_commands_.AddCommand(&testCorrection_);

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

  
  digitalWrite(motors_en_pin, LOW); // enable steppers

  // Start bluetooth
  SerialBT.begin("ESP32test2"); //Bluetooth device name

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
    spinBase(cw, false);   
  }

  serial_commands_.ReadSerial();

  // Handle bluetooth connection
  if(SerialBT.hasClient() == 0)
  {
    // Serial.println("No client connected");
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
        case 's':
          scramble(NULL);
          break;
        case 'z':
          zenMode(NULL);
          break;
        case 'r':
          speeen(NULL);
          break;
        case 't':
          toggleSteppers(NULL);
          break;     

        // Cubing notation moves
        case 'b':
          moveArmTo(MIDDLE); 
          closeHand();
          spinBase(cw, true);
          openHand();
          break;

        case 'B':
          moveArmTo(MIDDLE); 
          closeHand();
          spinBase(ccw, true);
          openHand();
          break;

        case 'y':
          spinBase(ccw, true);
          break;

        case 'Y':
          spinBase(cw, true);
          break;

        case 'X':
          flipCube();
          break;



        default:
          if(BluetoothIn != '\r' && BluetoothIn != '\n')
          {
            Serial.write("Received unknown bluetooth command: ");
            Serial.write(BluetoothIn);
            Serial.println();
          }
      }
    }
  }

  delay(1);
}
