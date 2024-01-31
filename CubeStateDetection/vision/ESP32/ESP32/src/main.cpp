#include <SerialCommands.h>
#include "BluetoothSerial.h"
#include <Adafruit_INA219.h>
#include <ArduinoSort.h>

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
#define OPENED           1
#define OPEN_AT_ENDSTOP  2
#define MIDDLE         420
#define DROPOFF_HEIGHT  69
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

Adafruit_INA219 lightRingINA; // current sensor
Adafruit_INA219 gripperINA(0x041); //gripper current sensor
typedef struct{
  float*  data;
  int   length;
}Array;


#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled, run `make menuconfig` to and enable it
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_SPEED 3.3        // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
#define MIN_SPEED 0.000001   // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
///////////////////////////////////////////////////////////////////////////////////////////////////////////
int gripStrength                =     388;
int moveArmSpeed                =      85;        // set the velocity (1-100) that we will raise or lower the arm
int handOpenCloseSpeed          =      20;  // set the velocity (1-100) that we will open and close the ha
int spinSpeed                   =     150;
int betweenActionsDelay         =      10;
int cubeDropDistance            =     400;
int numStepsFromBottomToMiddle  =     600;
int numStepsFromTopToMiddle     =    1350;
int numStepsFromDropoffToMiddle =     850;
int numStepsTopToBottom         =       0;

float cubeRotationError         =       10; // FLAG - This is currently set for Bruno's cube. Whatever this number is for other cubes needs to be calculated using comp. vision
int correctionSpeed             =      20;

int homePosition                =  MIDDLE;
int zenSpinSpeed                =      50;
int zenArmSpeed                 =      10;
int zenHandOpenCloseSpeed       =      10;
int faceRotationErrorCounter    =       0;
int numRotationsB4SecondaryCorrection = 2;
float fixCubeDegrees            =      45;
double minimumBaseVelocity      =    10.0;
int firstTimeOpening            =       1;

int armSpeedupDenominator       =       5; // this means that for the first 1/armSpeedupDenominator the arm will move slowly,
                                           // and for the last (armSpeedupDenominator - 1)/armSpeedupDenominator percentage of the move it will slow down
                                           // ie, if this is 4, the first 25% of the move will be accelerating (linearly) and the last 25% will be decelerating 
                                           // in other words, we'll be at the 'posted' speed 50% of the time.
int handSpeedupDenominator      =       5; // see above
int spinSpeedupDenominator      =       5; // see above
double minimumArmSpeed          =      25; // when we calculate an acceleration on the above lines, the first step delays will be too long (movement will be too slow)
double minimumHandSpeed         =      10; // by setting these, we can force the initial speed of the arm/hand/base spinning to some values
double minimumSpinSpeed         =      60;
//////////////////////////////////////////////////////// calculations for absolute movement (mm) calculations
double armStepsPerMm        = 1900.0/45.0;
double cubeletLength            =    17.0; // in mm
double bottomEndstopHeight      =    46.8; // this is the distance in mm between the base and the top edge of the hand
double topEndstopHeight         =    94.5; // in mm (measured from the base to the top edge of the hand)
double bottomGripHeight         =    bottomEndstopHeight + 2.0; 
double middleHeight             =    bottomGripHeight + cubeletLength - 2.0;
double topOfRotationHeight      =    bottomGripHeight + 2 * cubeletLength + 12.0;
double dropoffHeight            =    topOfRotationHeight - 9.0;
double currentCubeHeight        =    0;
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
void spinBase(int,bool);
void toggleSteppers(SerialCommands *sender);
void moveArmToMM(int destination); 

char serial_command_buffer_[32];
SerialCommands serial_commands_(&Serial, serial_command_buffer_, sizeof(serial_command_buffer_), "\r\n", " ");
void cmd_unrecognized(SerialCommands* sender, const char* cmd) {
sender->GetSerial()->println("Unrecognized command");}

void  homeArmAndHand() {
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
  moveArmToMM(homePosition);
  }}
void  centreLight(){
  float maxCurrent;
  float tempCurrent;
  int spinCount = 0;

  lightRingINA.powerSave(false);

  maxCurrent = lightRingINA.getCurrent_mA();

  for(int i = 1; i < 4; i++){
    spinBase(cw, false);

    tempCurrent = lightRingINA.getCurrent_mA();

    if(tempCurrent > maxCurrent){
      maxCurrent = tempCurrent;
      spinCount = i;
    }

    Serial.printf("Max Current: %f,\tCurrent Measured: %f\r\n", maxCurrent, tempCurrent);
    Serial.printf("Light Position: %d,\t\tCurrent Position: %d\r\n", spinCount, i);  
  }

  if(spinCount < 3){
    for(int i = 0; i <= spinCount; i++){
      spinBase(cw,false);
    }
  }

  lightRingINA.powerSave(true);}
void  homeLight(){
  float maxCurrent;
  float tempCurrent;
  int lightRingPos = 0;
  const float SAMPLE_NUM = 1280;  // number of times to sample the current

  int stepDelay = getDelay(25);

  float stepsPerSample = 19200 / SAMPLE_NUM;

  lightRingINA.powerSave(false);

  maxCurrent = lightRingINA.getCurrent_mA();

  // Serial.printf("Max Current: %f,\tCurrent Measured: %f\r\n", maxCurrent, maxCurrent);
  // Serial.printf("Light Position: %d,\t\tCurrent Position: %d\r\n", lightRingPos, 0);

  digitalWrite(motors_base_dir_pin, cw);  // set the direction

  for(int i = 1; i < SAMPLE_NUM; i++){
    tempCurrent = lightRingINA.getCurrent_mA();

    // spin base
    for (int i = 0; i < stepsPerSample; i++) {
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
      delayMicroseconds(stepDelay);
    }

    if(tempCurrent > maxCurrent){
      maxCurrent = tempCurrent;
      lightRingPos = i;
    }

    Serial.printf("Max Current: %f,\tCurrent Measured: %f\r\n", maxCurrent, tempCurrent);
    Serial.printf("Light Position: %d,\t\tCurrent Position: %d\r\n", lightRingPos, i);
  }

  if(lightRingPos < SAMPLE_NUM/2){
    digitalWrite(motors_base_dir_pin, cw);
    for(int i = 0; i <= lightRingPos; i++){
        // spin base
        for (int i = 0; i < stepsPerSample; i++) {
          digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
          delayMicroseconds(stepDelay);
        }
    }
  }
  else if (lightRingPos < SAMPLE_NUM - 1){ // needs some better math
    digitalWrite(motors_base_dir_pin, ccw);  // set the direction

    // Serial.println("Difference %d", SAMPLE_NUM - lightRingPos);

    for(int i = 0; i < SAMPLE_NUM - lightRingPos; i++){
        // spin base
        for (int i = 0; i < stepsPerSample; i++) {
          digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
          delayMicroseconds(stepDelay);
        }
    }
  }

  lightRingINA.powerSave(true);}
void  homeLightv2(){
  float tempCurrent;
  // const float thresh = 100;
  const int SAMPLE_NUM = 256*5;  // number of times to sample the current should be a factor of motor steps

  float samples[SAMPLE_NUM];

  int stepDelay = getDelay(25);

  int stepsPerSample = 19200 / SAMPLE_NUM;

  float gaussian[3] = {1/4, 2/4, 1/4};

  lightRingINA.powerSave(false);

  digitalWrite(motors_base_dir_pin, cw);  // set the direction

  for(int i = 0; i < SAMPLE_NUM; i++){

     samples[i] = 0;

    for(int j = 0; j < 5; j++){
      tempCurrent = lightRingINA.getCurrent_mA();

      samples[i] += tempCurrent;

      delayMicroseconds(100);
    }

    samples[i] /= 5;

    // tempCurrent = lightRingINA.getCurrent_mA();

    // spin base
    for (int i = 0; i < stepsPerSample; i++) {
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
      delayMicroseconds(stepDelay);
    }

    // samples[i] = tempCurrent;

    // Serial.printf("Max Current: %f,\tCurrent Measured: %f\r\n", maxCurrent, tempCurrent);
    // Serial.printf("Light Position: %d,\t\tCurrent Position: %d\r\n", lightRingPos, i);
  }

  Serial.print("[");

  for(int i = 0; i < SAMPLE_NUM; i++){
    Serial.printf("%f ", samples[i]);
  }

  Serial.println("]");



  // if(lightRingPos < SAMPLE_NUM/2-1){
  //   digitalWrite(motors_base_dir_pin, cw);  // set the direction
  //   for(int i = 0; i <= lightRingPos; i++){
  //       // spin base
  //       for (int i = 0; i < stepsPerSample; i++) {
  //         digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
  //         delayMicroseconds(stepDelay);
  //       }
  //   }
  // }
  // else if (lightRingPos < SAMPLE_NUM - 1){
  //   digitalWrite(motors_base_dir_pin, ccw);  // set the direction
  //   for(int i = 0; i < lightRingPos; i++){
  //       // spin base
  //       for (int i = 0; i < stepsPerSample; i++) {
  //         digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
  //         delayMicroseconds(stepDelay);
  //       }
  //   }
  // }

  lightRingINA.powerSave(true);}
void  smallScan(){
  const int STEPS_LR = 19200/360*12;

  float stepDelay = getDelay(100);

  float lightPos = 0;
  float currVal = 0;

  int stepDelta = 0;

  float samples[2*STEPS_LR + 1];

  int totalSampoles = 1;
  int boxcarWidth = 5;
  lightRingINA.powerSave(false);

  delay(500);

  for(int j = 0; j < boxcarWidth; j++){
    currVal += lightRingINA.getCurrent_mA();
    delayMicroseconds(200);
  }

  currVal /= 5;

  samples[STEPS_LR] = currVal;

  digitalWrite(motors_base_dir_pin, cw);

  // move to end of scan area
  for(int i = 0; i < STEPS_LR; i++){
    // spin base
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
    delayMicroseconds(stepDelay);
    stepDelta--;
  }

  digitalWrite(motors_base_dir_pin, ccw);

  // scan for best loation
  for(int i = 0; i < STEPS_LR*2+1; i++){
    // reset value
    currVal = 0;
    stepDelta++;

    // spin base
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
    delayMicroseconds(10);

    for(int j = 0; j < 5; j++){
      currVal += lightRingINA.getCurrent_mA();
      delayMicroseconds(200);
    }

    currVal /= 5;

    samples[i] = currVal;
  }

  // reset location
  digitalWrite(motors_base_dir_pin, cw);

  for(int i = 0; i < STEPS_LR+1; i++){
    // spin base
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));  // Perform one motor step
    delayMicroseconds(stepDelay);
    stepDelta--;
  }

  // print data (remove later)
  // Serial.print("[");

  // for(int i = 0; i < 2*STEPS_LR+1; i++){
  //   Serial.printf("%f ", samples[i]);
  // }

  // Serial.println("]");

  Serial.println(stepDelta);

  lightRingINA.powerSave(true);}
void  scanLightCurrent(SerialCommands *sender){
  int threshold = 110;
  Serial.println("How many times to cross current threshold?");
  int numTimesToCross = getIntegerFromUser();
  int numTimesCrossed = 0;
  lightRingINA.powerSave(false);
  digitalWrite(motors_base_dir_pin, ccw);
  float currentMeasurement;
  Serial.println("How large should the boxcar be?");
  int boxCarLength = getIntegerFromUser(); // boxcar size must be odd
  float sectionMeasurements[boxCarLength];
  float iterationMedian;
  float thisMeasurement;

  while(!Serial.available()){
    for(int j = 0; j < boxCarLength; j++){
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      thisMeasurement = lightRingINA.getCurrent_mA();      
      sectionMeasurements[j] = thisMeasurement;  
      delay(2);        
    }  
    sortArray(sectionMeasurements, boxCarLength);
    iterationMedian = sectionMeasurements[(boxCarLength + 1 ) / 2 - 1];
    Serial.println(iterationMedian);  
    if(iterationMedian >= threshold){
      numTimesCrossed++;
    }
    if(numTimesCrossed > numTimesToCross){
      return;
    }
  }}
void  homeBase(){
  int threshold = 110;
  Serial.println("Homing base");
  int numTimesToCross = 16;
  int numTimesCrossed = 0;
  lightRingINA.powerSave(false);
  digitalWrite(motors_base_dir_pin, ccw);
  float currentMeasurement;
  int boxCarLength = 27; // boxcar size must be odd
  float sectionMeasurements[boxCarLength];
  float iterationMedian;
  float thisMeasurement;

  while(numTimesCrossed < numTimesToCross){
    for(int j = 0; j < boxCarLength; j++){
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      thisMeasurement = lightRingINA.getCurrent_mA();      
      sectionMeasurements[j] = thisMeasurement;  
      delay(1);        
    }  
    sortArray(sectionMeasurements, boxCarLength);
    iterationMedian = sectionMeasurements[(boxCarLength + 1 ) / 2 - 1];
    Serial.println(iterationMedian);  
    if(iterationMedian >= threshold){
      numTimesCrossed++;
    }  
  }}
void  homeBase2(){
    lightRingINA.powerSave(false);
      Serial.println("would you like to scan or have a shot at it? (0 to scan, 1 to try)");
      int mode = getIntegerFromUser();  
      Serial.println("What should the cutoff threshold be?");
      int threshold = getIntegerFromUser();
    int queueLength = 289;
    float currentReading;
    float queue[queueLength];
    float tempQueue[queueLength];
    float median = 0;
    int graphBottom = 70;
    int graphTop   = 120;
    int divisor = 1;
    int timesCrossed = 0;
    int tic;
    int toc;
   

    for(int i = 0; i < queueLength; i++){
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      currentReading = lightRingINA.getCurrent_mA();
      queue[i] = currentReading;  // fill the queue
      delayMicroseconds(600);
    }
    memcpy(&tempQueue, &queue, sizeof(queue));    
    sortArray(tempQueue, queueLength);
    median = tempQueue[(queueLength + 1 ) / 2 - 1];
    Serial.println(median);

    switch(mode){
      case 0:        
        while(!Serial.available()){
          tic = micros();
          for(int i = 0; i < queueLength - 1; i++){
            queue[i] = queue[i+1]; // update the queue
          }
          digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
         
          queue[queueLength - 1] = lightRingINA.getCurrent_mA();// done updating queue

          memcpy(&tempQueue, &queue, sizeof(queue));    //fill the temp queue
          sortArray(tempQueue, queueLength);
          median = tempQueue[(queueLength + 1 ) / 2 - 1]; // see what the median is
          if(divisor % 11 == 0){
            if(median >= threshold){
              timesCrossed++;
            }
            if(timesCrossed == 41){
              return;
            }
            divisor = 1;
            Serial.print(graphBottom);
            Serial.print(" ");
            Serial.print(graphTop);
            Serial.print(" ");
            Serial.println(median);                
           
          }else{
            divisor++;
          }
          toc = micros() - tic;
          delayMicroseconds(600);
        }
        break;

        case 1:
          while(median < threshold){
          for(int i = 0; i < queueLength - 1; i++){
            queue[i] = queue[i+1]; // update the queue
          }
          digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
          queue[queueLength - 1] = lightRingINA.getCurrent_mA();

          memcpy(&tempQueue, &queue, sizeof(queue));    
          sortArray(tempQueue, queueLength);
          median = tempQueue[(queueLength + 1 ) / 2 - 1];
          Serial.println(median);
          delayMicroseconds(600);
        }          
          break;
    }}
void  homeBase3(){
    lightRingINA.powerSave(false);
    int shortDelay = 0;
    int longDelay  = 600;
    int stepDelay = longDelay;
    int queueLength = 289;
    float currentReading;
    float queue[queueLength];
    float tempQueue[queueLength];
    float median     = 0;
    int divisor      = 1;
    int timesCrossed = 0;  
    int threshold  = 110;

    for(int i = 0; i < queueLength; i++){
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      currentReading = lightRingINA.getCurrent_mA();
      queue[i] = currentReading;  // fill the queue
      delayMicroseconds(shortDelay);
    }
    memcpy(&tempQueue, &queue, sizeof(queue));    
    sortArray(tempQueue, queueLength);
    median = tempQueue[(queueLength + 1 ) / 2 - 1];    
         
        while(timesCrossed < 40){

          // this for loop should be a function
          for(int i = 0; i < queueLength - 1; i++){
            queue[i] = queue[i+1]; // update the queue
          }

          digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));          
          queue[queueLength - 1] = lightRingINA.getCurrent_mA();// done updating queue

          // these three lines of code should also be a function
          memcpy(&tempQueue, &queue, sizeof(queue));    //fill the temp queue
          sortArray(tempQueue, queueLength);
          median = tempQueue[(queueLength + 1 ) / 2 - 1]; // see what the median is
         
          if(median > 105)
          {
            // Serial.println("Set long delay as we approach our destination");
            stepDelay = longDelay;            
          }
          else
          {
            // Serial.println("we are far. Setting a short travel delay.");
            stepDelay = shortDelay;
          }
          if(divisor % 13 == 0){
            if(median >= threshold){
              timesCrossed++;
            }else if(median < 90){
              timesCrossed = 0;
              stepDelay = shortDelay;
              // Serial.println("Set short delay at divisor check.");
            }        
            divisor = 1;
          }else{
            divisor++;
          }        
          // Serial.print("Delaying for "); Serial.println(stepDelay); Serial.println(" microseconds.");
          delayMicroseconds(stepDelay);
        }    
          // toggling the steppers off and on will force the motor to settle at the nearest pole.
          //Assuming we're pretty close, we'll get to the exact center.
        toggleSteppers(NULL);
        delay(100);
        toggleSteppers(NULL);
    }
void  homeBase4(){
    lightRingINA.powerSave(false);
    int shortDelay = 10;
    int longDelay  = 600;
    int stepDelay = longDelay;
    int queueLength = 97;
    float currentReading;
    float queue[queueLength];
    float tempQueue[queueLength];
    float median     = 0;
    int divisor      = 1;
    int timesCrossed = 0;  
    int threshold  = 110;

    for(int i = 0; i < queueLength; i++){
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      currentReading = lightRingINA.getCurrent_mA();
      queue[i] = currentReading;  // fill the queue
      delayMicroseconds(shortDelay);
    }
    memcpy(&tempQueue, &queue, sizeof(queue));    
    sortArray(tempQueue, queueLength);
    median = tempQueue[(queueLength + 1 ) / 2 - 1];    
         
        while(timesCrossed < 41){

          // this for loop should be a function
          for(int i = 0; i < queueLength - 1; i++){
            queue[i] = queue[i+1]; // update the queue
          }

          digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));          
          queue[queueLength - 1] = lightRingINA.getCurrent_mA();// done updating queue

          // these three lines of code should also be a function
          memcpy(&tempQueue, &queue, sizeof(queue));    //fill the temp queue
          sortArray(tempQueue, queueLength);
          median = tempQueue[(queueLength + 1 ) / 2 - 1]; // see what the median is
         
          if(median > 105)
          {
            // Serial.println("Set long delay as we approach our destination");
            stepDelay = longDelay;            
          }
          else
          {
            // Serial.println("we are far. Setting a short travel delay.");
            stepDelay = shortDelay;
          }
          if(divisor % 13 == 0){
            if(median >= threshold){
              timesCrossed++;
            }else if(median < 90){
              timesCrossed = 0;
              stepDelay = shortDelay;
              // Serial.println("Set short delay at divisor check.");
            }        
            divisor = 1;
          }else{
            divisor++;
          }        
          // Serial.print("Delaying for "); Serial.println(stepDelay); Serial.println(" microseconds.");
          delayMicroseconds(stepDelay);
        }    
          // toggling the steppers off and on will force the motor to settle at the nearest pole.
          //Assuming we're pretty close, we'll get to the exact center.
        toggleSteppers(NULL);
        delay(100);
        toggleSteppers(NULL);
    }
void  homeBase5(){
    lightRingINA.powerSave(false);
    int shortDelay = 0;
    int longDelay  = 600;
    int stepDelay = longDelay;
    int queueLength = 49;
    float currentReading;
    float queue[queueLength];
    float tempQueue[queueLength];
    float median     = 0;
    int divisor      = 1;
    int timesCrossed = 0;  
    int threshold  = 110;

    for(int i = 0; i < queueLength; i++){
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      currentReading = lightRingINA.getCurrent_mA();
      queue[i] = currentReading;  // fill the queue
      delayMicroseconds(shortDelay);
    }
    memcpy(&tempQueue, &queue, sizeof(queue));    
    sortArray(tempQueue, queueLength);
    median = tempQueue[(queueLength + 1 ) / 2 - 1];    
         
        while(timesCrossed < 41){
          // this for loop should be a function
          for(int i = 0; i < queueLength - 1; i++){
            queue[i] = queue[i+1]; // update the queue
          }

          digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));          
          queue[queueLength - 1] = lightRingINA.getCurrent_mA();// done updating queue

          // these three lines of code should also be a function
          memcpy(&tempQueue, &queue, sizeof(queue));    //fill the temp queue
          sortArray(tempQueue, queueLength);
          median = tempQueue[(queueLength + 1 ) / 2 - 1]; // see what the median is
         
          if(median > 105)
          {
            // Serial.println("Set long delay as we approach our destination");
            stepDelay = longDelay;            
          }
          else
          {
            // Serial.println("we are far. Setting a short travel delay.");
            stepDelay = shortDelay;
          }
          if(divisor % 13 == 0){
            if(median >= threshold){
              timesCrossed++;
            }else if(median < 90){
              timesCrossed = 0;
              stepDelay = shortDelay;
              // Serial.println("Set short delay at divisor check.");
            }        
            divisor = 1;
          }else{
            divisor++;
          }        
          // Serial.print("Delaying for "); Serial.println(stepDelay); Serial.println(" microseconds.");
          delayMicroseconds(stepDelay);
        }    
          // toggling the steppers off and on will force the motor to settle at the nearest pole.
          //Assuming we're pretty close, we'll get to the exact center.
        toggleSteppers(NULL);
        delay(100);
        toggleSteppers(NULL);
    }

void  copy(float* src, float* dst, int len) {
    memcpy(dst, src, sizeof(src[0])*len);}
void  microScan(){
  lightRingINA.powerSave(false);
  int degreesToSweep = 24;
  int boxCarLength   = 7; // boxcar size must be odd
  int sweepSize      = 19200 / 360 * degreesToSweep;
  int currrentVal;
  float sectionMeasurements[boxCarLength];
  float iterationMedian;
  float thisMeasurement;
  while((sweepSize % boxCarLength) > 0){ // increase the size of the sweep until the number of steps is a multiple of the boxcar length
    sweepSize++;
  }
  int numberOfMedianMeasurements = sweepSize/boxCarLength;
  float mediansArray[numberOfMedianMeasurements];

  digitalWrite(motors_base_dir_pin, ccw);
  for(int i = 0; i < numberOfMedianMeasurements; i++){

    for(int j = 0; j < boxCarLength; j++){
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      thisMeasurement = lightRingINA.getCurrent_mA();
      Serial.println(thisMeasurement);
      sectionMeasurements[j] = thisMeasurement;  
      delay(2);        
    }  
    sortArray(sectionMeasurements, boxCarLength);
    iterationMedian = sectionMeasurements[(boxCarLength + 1 ) / 2 - 1];
    //Serial.println(iterationMedian);
    mediansArray[i] = iterationMedian;
  }}
Array* movingMedian(float* samples, int length, int windowSize){
  int arrLength = length / windowSize;

  float median;

  Array* arr = (Array *)malloc(sizeof(Array));

  arr->length = arrLength;
  arr->data   = (float *)malloc(arrLength*sizeof(float));

  // actually average oops... I'll fix this later
  for(int i = 0; i < arr->length; i++){
    // reset median
    median = 0;
    for(int j = i*windowSize; j < (i+1)*windowSize; j++){
      median += samples[j];
    }
    median /= windowSize;

    arr->data[i] = median;
  }

  return arr;}

int   findLightRing(float* samples, int length){
  Array* arr = movingMedian(samples, length, 5);

  const float TOL = 0.5;

  int streakStart = 0;
  int tempStart = 1;
  int streakLen = 0;
  int tempLen = 1;

  int lightPos;

  for(int i = 1; i < arr->length; i++){
    if(abs(arr->data[i] - arr->data[i-1]) < TOL){
      tempLen++;
    }
    else{
      if(tempLen > streakLen){
        streakLen = tempLen;
        streakStart = tempStart;
      }

      tempStart = i;
      tempLen = 1;
    }
  }

  if(tempLen > streakLen){
    streakLen = tempLen;
    streakStart = tempStart;
  }

  lightPos = 5*(2*streakStart/length + 1);

  return lightPos;}
void  moveArmTo(int destination) {
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
    case DROPOFF_HEIGHT:
      Serial.print("dropoff height");
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
    case DROPOFF_HEIGHT:
      desiredLocation = "dropoff height";
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
      while (digitalRead(BOTTOM) == 1){          
        moveArm(DOWN);
      }
      armLocation = BOTTOM;
      break;
    /////////////////////////////////////////////////////////
    case TOP:
      while(digitalRead(TOP) == 1){   
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
      else if (armLocation == DROPOFF_HEIGHT){
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
        delay(betweenActionsDelay);
        for(int i = 0; i < numStepsFromTopToMiddle; i++){
              moveArm(DOWN);            
        }
      }
        armLocation = MIDDLE;
        break;    
  }
    Serial.println("Arrived at destination.");
    delay(betweenActionsDelay);}
void  closeHand() {
  digitalWrite(motors_arm_left_dir_pin, ccw);
  digitalWrite(motors_arm_right_dir_pin, ccw);
  Serial.println("Closing hand");
  int point1 = gripStrength / handSpeedupDenominator;
  int point2 = gripStrength * (handSpeedupDenominator - 1) / handSpeedupDenominator;
  double velocity;
  int stepDelay;
  int gs = gripStrength;

  if(handState == UNKNOWN)
    openHand();

  if(handState == OPENED)
    gs -= 100;


  for (int i = 0; i < gs; i++) {        
      if(i < point1){
      velocity = handOpenCloseSpeed * double(i)/point1;        
      }
      else if(i >= point1 && i <= point2){
      velocity = handOpenCloseSpeed;
    }else{
      velocity = handOpenCloseSpeed * (gripStrength - double(i)) / point1;
    }
      velocity  = max(velocity, minimumHandSpeed);
      stepDelay = getDelay(velocity);  

      digitalWrite(motors_arm_left_step_pin,  !digitalRead(motors_arm_left_step_pin));    // Step left motor
      digitalWrite(motors_arm_right_step_pin, !digitalRead(motors_arm_right_step_pin));  // Step right motor   
      delayMicroseconds(stepDelay);
  }
  
  delay(betweenActionsDelay);
  handState = CLOSE; }  
void  openHand() {

  Serial.println("Opening hand");
  int count = 0;

  if(handState == OPENED || handState == OPEN_AT_ENDSTOP){
    Serial.println("Hand was already opened.");
    return;
  }

  if(handState == UNKNOWN ){
    Serial.println("Didn't know where hand was- opening hand until we hit the end stop.");
    while(digitalRead(endstop_arm_openLimit_pin) == 1){
      count++;
      articulateHand(OPEN);
      int stepDelay = getDelay(handOpenCloseSpeed);
      delayMicroseconds(stepDelay);
      if (count > 1000){
        gripperFunctional = false;
        Serial.println("Hand could not reach end stop");
        break;
      }
    }    
    handState = OPEN_AT_ENDSTOP; 
  }
  if(handState == CLOSED){ // if the hand is closed, we'll move the hand open for gripStrength - 100 steps (to not hit the end stop)
    digitalWrite(motors_arm_left_dir_pin, cw);
    digitalWrite(motors_arm_right_dir_pin, cw);
    Serial.println("Opening hand without hitting endstop");
    int point1 = gripStrength / handSpeedupDenominator;
    int point2 = gripStrength * (handSpeedupDenominator - 1) / handSpeedupDenominator;
    double velocity;
    int stepDelay;
    int gs = gripStrength - 100;

    for (int i = 0; i < gs; i++) {        
       if(i < point1){
        velocity = handOpenCloseSpeed * double(i)/point1;        
       }
       else if(i >= point1 && i <= point2){
        velocity = handOpenCloseSpeed;
      }else{
        velocity = handOpenCloseSpeed * (gs - double(i)) / point1;
      }
       velocity  = max(velocity, minimumHandSpeed);
       stepDelay = getDelay(velocity);  

       digitalWrite(motors_arm_left_step_pin,  !digitalRead(motors_arm_left_step_pin));    // Step left motor
       digitalWrite(motors_arm_right_step_pin, !digitalRead(motors_arm_right_step_pin));  // Step right motor   
       delayMicroseconds(stepDelay);
   }
  handState = OPENED;
  }
  
  delay(betweenActionsDelay);}

void  spinBase(int my_direction, bool correctionEnabled) {   
  Serial.println("Spinning base once");
  int degreesToRotate = 90;  

  digitalWrite(motors_base_dir_pin, my_direction);  // set the direction
  
  if (armLocation != TOP && armLocation != MIDDLE && gripperFunctional) {
    openHand();
    moveArmToMM(MIDDLE);
  }

  if(correctionEnabled){
    degreesToRotate += cubeRotationError;
  }

  double totalSteps = 19200.0 * degreesToRotate / 360.0;
  int actualSteps = floor(totalSteps);
  int stepDelay = 0;
  double velocity;
  float point1 = totalSteps / spinSpeedupDenominator;
  float point2 = totalSteps * (spinSpeedupDenominator - 1) / spinSpeedupDenominator;

  for (int i = 0; i < actualSteps; i++) {        
       if(i < point1){
        velocity = spinSpeed * double(i)/point1;        
       }
       else if(i >= point1 && i <= point2){
        velocity = spinSpeed;
      }else{
        velocity = spinSpeed * (totalSteps - double(i)) / point1;
      }
       velocity  = max(velocity, minimumSpinSpeed);
       stepDelay = getDelay(velocity);  

       digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));     
       delayMicroseconds(stepDelay);
  }

  if(correctionEnabled){
    digitalWrite(motors_base_dir_pin, !digitalRead(motors_base_dir_pin));  // change the direction  
    int correctionStepDelay = getDelay(correctionSpeed);                   // set the movement speed for the alignment
    openHand();                                                            // stop carrying the cube
    for(int i = 0; i < cubeRotationError * 19200.0 / 360.0; i++){           // undo the base error
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
    delayMicroseconds(correctionStepDelay);
  }

  }  
    delay(betweenActionsDelay);
    Serial.println("Done spinning base once");}
void  spinBaseTwice(bool correctionEnabled){

  Serial.println("Spinning base twice");
  int degreesToRotate = 180;  
  
  if (armLocation != TOP && armLocation != MIDDLE && gripperFunctional) {
    openHand();
    moveArmToMM(MIDDLE);
  }

  if(correctionEnabled){
    degreesToRotate += cubeRotationError;
  }

  double totalSteps = 19200.0 * degreesToRotate / 360.0;
  int actualSteps = floor(totalSteps);
  int stepDelay = 0;
  double velocity;
  float point1 = totalSteps    /(spinSpeedupDenominator * 2);
  float point2 = totalSteps * (spinSpeedupDenominator * 2 - 1) / (spinSpeedupDenominator * 2);

  for (int i = 0; i < actualSteps; i++) { 
       
       if(i < point1){
        velocity = spinSpeed * double(i)/point1;        
       }
       else if(i >= point1 && i <= point2){
        velocity = spinSpeed;
      }else{
        velocity = spinSpeed * (totalSteps - double(i)) / point1;
      }
       velocity  = max(velocity, minimumSpinSpeed);
       stepDelay = getDelay(velocity);   

       digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));    
       delayMicroseconds(stepDelay);
  }

  if(correctionEnabled){
    digitalWrite(motors_base_dir_pin, !digitalRead(motors_base_dir_pin));  // change the direction  
    int correctionStepDelay = getDelay(correctionSpeed);                   // set the movement speed for the alignment
    openHand();
    for(int i = 0; i < cubeRotationError * 19200.0 / 360.0; i++){           // undo the base error
      digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
      delayMicroseconds(correctionStepDelay);
    }
  }
  delay(betweenActionsDelay);
  Serial.println("Done spinning base twice");}
float getFloatFromUser(){
  Serial.println("Please enter a value:");
  while (!Serial.available()) {}  // wait until the user enters a value
  float value = Serial.parseFloat();
  while (Serial.available()) {  // flush the serial input buffer
    int flush = Serial.read();
  }
  return value;}
void  moveArm(int direction) {
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
  }
void  articulateHand(int direction) {
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
}
int   getDelay(int velocity) {
  velocity = min(velocity, 200);
  double x = MIN_SPEED + velocity * (MAX_SPEED - MIN_SPEED) / 100;
  double delayDuration = pow(0.0003 * x, -1) / 10;
  return round(delayDuration);}
int   getIntegerFromUser() {
  Serial.println("Please enter a value:");
  while (!Serial.available()) {}  // wait until the user enters a value
  int value = Serial.parseInt();
  while (Serial.available()) {  // flush the serial input buffer
    int flush = Serial.read();
  }
  return value;}
void  fixCubeError(SerialCommands *sender){
  int stepDelay = getDelay(spinSpeed);
  openHand();
  moveArmToMM(MIDDLE);
  closeHand();

  for(int i = 0; i < fixCubeDegrees * 19200.0 / 360.0; i++){ //rotate cube x degrees. This squashes error in other faces
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
    delayMicroseconds(stepDelay);
  }
  delay(betweenActionsDelay); 

  digitalWrite(motors_base_dir_pin, !digitalRead(motors_base_dir_pin)); // change direction 


  for(int i = 0; i <  (fixCubeDegrees + cubeRotationError) * 19200.0 / 360.0; i++){// Rotate back, moving the base past center, but aligning the cube with itself, adding one rotation error
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
    delayMicroseconds(stepDelay);
  }
  delay(betweenActionsDelay); 

  digitalWrite(motors_base_dir_pin, !digitalRead(motors_base_dir_pin)); // change direction)
   
  openHand();
  for(int i = 0; i < cubeRotationError * 19200.0 / 360.0; i++){ // undo the base error
    digitalWrite(motors_base_step_pin, !digitalRead(motors_base_step_pin));
    delayMicroseconds(stepDelay);
  }

  delay(betweenActionsDelay);  }
void  flipCube() { 
  Serial.println("////// Flipping cube");
  openHand();
  moveArmToMM(BOTTOM);
  closeHand();
  moveArmToMM(TOP);
  delay(100);
  moveArmToMM(DROPOFF_HEIGHT);
  openHand();}
void  rotateFace(int face, int singleOrDouble) {
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
      spinBaseTwice(false);        
      flipCube();
      break;
    case REARCCW:
      Serial.println("/////////////// Turning rear face counterclockwise");
      direction = cw;
      spinBaseTwice(false);        
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
      fixCubeError(NULL);
      flipCube();
      break;
    case TOPCCW:
        Serial.println("/////////////// Turning top face counterclockwise");
        direction = ccw;
        flipCube();
        fixCubeError(NULL);
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

  moveArmToMM(MIDDLE);
  if(handState != CLOSED)
    closeHand();
 
  if(singleOrDouble == 2){
    spinBaseTwice(true);
  }else{
    spinBase(direction, true);
  }

  openHand();
  armLocation = MIDDLE;
  handState   = OPENED;
  delay(betweenActionsDelay);
  faceRotationErrorCounter++;
  }

void  readCurrent(){
  lightRingINA.powerSave(false);

  float avg = 0;

  for(int i = 0; i < 1000; i++){
    avg += lightRingINA.getCurrent_mA();
  }

  avg /= 1000;

  Serial.printf("Current is %fmA\r\n", avg);}
void  testCorrection(SerialCommands *sender){
  moveArmToMM(MIDDLE);
  closeHand();
  Serial.println("How many degrees to correct?");
  cubeRotationError = getIntegerFromUser();
  spinBase(cw, true);
 }
void  testDistanceToMiddleFromTop(SerialCommands * sender) {
  int value = getIntegerFromUser();
  moveArmToMM(TOP);
  for (int i = 0; i < value; i++) {
    moveArm(DOWN);
    int stepDelay = getDelay(moveArmSpeed);
    delayMicroseconds(stepDelay);
  }}
void  testDistanceToMiddleFromBottom(SerialCommands * sender) {
  int value = getIntegerFromUser();
  moveArmToMM(BOTTOM);
  for (int i = 0; i < value; i++) {
    moveArm(UP);
    int stepDelay = getDelay(moveArmSpeed);
    delayMicroseconds(stepDelay);
  }}
void  setDistanceToMiddleFromCubeRelease(SerialCommands *sender){
  openHand();
  moveArmToMM(TOP);
  for(int i = 0; i < cubeDropDistance; i++){
    moveArm(DOWN);
    int stepDelay = getDelay(moveArmSpeed);
    delayMicroseconds(stepDelay);
  }
  Serial.println("Please enter how many steps to travel from the dropoff location to the middle of the cube: ");
  int numSteps = getIntegerFromUser();

  for(int i = 0; i < numSteps; i++){
    moveArm(DOWN);
    int stepDelay = getDelay(moveArmSpeed);
    delayMicroseconds(stepDelay);
  }
  Serial.println("Set distance from dropoff to middle");
  numStepsFromDropoffToMiddle = numSteps; }
void  setbetweenActionsDelay(SerialCommands *sender){
  Serial.println("Please enter an inter-action delay (ms)");
  betweenActionsDelay = getIntegerFromUser();}
void  setScrambleAndSolveSpeed(SerialCommands *sender){
  Serial.println("Please enter the cube spin speed (0 - 120):");
  spinSpeed  = getIntegerFromUser();
  Serial.print("Set cube spin speed to "); Serial.println(spinSpeed);

  Serial.println("Please enter the arm move speed (0 - 100):");
  moveArmSpeed   = getIntegerFromUser();  
  Serial.print("Set arm speed to "); Serial.println(moveArmSpeed);

  Serial.println("Please enter the hand open/close speed (0 - 100):");
  handOpenCloseSpeed   = getIntegerFromUser();  
  Serial.print("Set hand speed to "); Serial.println(handOpenCloseSpeed);}
void  setCubeError(SerialCommands *sender){
  int errorAcceptable = 0;
  float value; 
  
  while(errorAcceptable == 0){
    Serial.println("Please make sure the cube is aligned and on the base. Enter anything to continue.");
    int dummy = getIntegerFromUser();
    moveArmToMM(MIDDLE);
    closeHand();
    Serial.println("Please set the cube rotation error in degrees (can be a decimal):");
    cubeRotationError = getFloatFromUser();
    spinBase(cw, true);
    openHand();
    Serial.println("Is the cube aligned? (1 = Yes, 0 = No)");
    errorAcceptable = getIntegerFromUser();
    if(errorAcceptable == 0)
      Serial.println("Please remember to align the cube before trying again-");
    }
  }
void  setGripDistance(SerialCommands * sender) {
  while (digitalRead(endstop_arm_openLimit_pin) == 1) {
    articulateHand(OPEN);
    int stepDelay = getDelay(handOpenCloseSpeed);
    delayMicroseconds(stepDelay);
  }
  Serial.println("Please enter how many steps to close the gripper for:");
  int numSteps = getIntegerFromUser();
  Serial.print("Now closing gripper for ");
  Serial.print(numSteps, DEC);
  Serial.println(" steps.");
  for (int i = 0; i < numSteps; i++) {
    articulateHand(CLOSE);
    int stepDelay = getDelay(handOpenCloseSpeed);
    delayMicroseconds(stepDelay);
  }
  gripStrength = numSteps;
  Serial.print("Set grip strength to ");
  Serial.println(numSteps);
  armLocation = UNKNOWN;
  handState = UNKNOWN;}
void  setZenSpeeds(SerialCommands * sender){
  Serial.println("Please enter the cube spin speed for zen mode (0 - 100):");
  zenSpinSpeed  = getIntegerFromUser();
  Serial.print("Set cube spin speed to "); Serial.println(zenSpinSpeed);

  Serial.println("Please enter the arm move speed for zen mode (0 - 100):");
  zenArmSpeed   = getIntegerFromUser();  
  Serial.print("Set arm speed to "); Serial.println(zenArmSpeed);

  Serial.println("Please enter the hand open/close speed for zen mode (0 - 100):");
  zenHandOpenCloseSpeed   = getIntegerFromUser();  
  Serial.print("Set arm speed to "); Serial.println(zenArmSpeed);}
void  toggleSteppers(SerialCommands *sender){
  digitalWrite(motors_en_pin,!digitalRead(motors_en_pin));
  Serial.println("Switched motors state (on/off)");}
void  readGripCurrent(SerialCommands *sender){

  openHand();
  moveArmToMM(MIDDLE);
  gripperINA.powerSave(false);
  float currentReading;
  float median = 0;
  int graphBottom = 500;
  int graphTop    = 700;   
  int potentialGripStrength = 0;
  float lastReading = 0;
  float thisReading = 0;
  
  for(int i = 0; i < 100; i++){
    articulateHand(CLOSE); potentialGripStrength++;
    lastReading = gripperINA.getCurrent_mA();
    delay(10);
    thisReading = gripperINA.getCurrent_mA(); 
    
  }

  while(true){ 
    
    articulateHand(CLOSE);   
    potentialGripStrength++;

    thisReading = gripperINA.getCurrent_mA(); 

            
      
    Serial.print(graphBottom);
    Serial.print(" ");
    Serial.print(graphTop);
    Serial.print(" ");
    Serial.print(thisReading);   
    Serial.print(" ");
    Serial.println(potentialGripStrength);
    delay(10);
    

    if (abs(thisReading - lastReading) > 50){
      Serial.println(potentialGripStrength);
      gripperINA.powerSave(true);
      handState = UNKNOWN;
      armLocation = UNKNOWN;
      return;
    } 
    lastReading = thisReading;
   
  }
    handState = UNKNOWN;
    armLocation = UNKNOWN;}
void  testNewCloseHand(SerialCommands *sender){
  moveArmToMM(MIDDLE);
  openHand();
  closeHand();}
void  measureDistanceTopToBottom(SerialCommands *sender){ 
  int stepDelay = getDelay(moveArmSpeed);
  openHand();
  moveArmToMM(TOP);

  while(digitalRead(endstop_arm_lowerLimit_pin) == 1 ){
    moveArm(DOWN); numStepsTopToBottom++;
    delayMicroseconds(stepDelay);
  }
  armLocation = BOTTOM;
  Serial.print("Measured "); Serial.print(numStepsTopToBottom);Serial.println(" steps from top to bottom.");}
void  moveArmMillimeters(double mm, int direction){
    double numSteps = round(armStepsPerMm * mm); 
    int point1 = numSteps / armSpeedupDenominator;
    int point2 = numSteps * (armSpeedupDenominator - 1) / armSpeedupDenominator;
    double velocity;
    int stepDelay;

    for (int i = 0; i < numSteps; i++) {        
       if(i < point1){
        velocity = moveArmSpeed * double(i)/point1;        
       }
       else if(i >= point1 && i <= point2){
        velocity = moveArmSpeed;
      }else{
        velocity = moveArmSpeed * (numSteps - double(i)) / point1;
      }
      velocity  = max(velocity, minimumArmSpeed);
      stepDelay = getDelay(velocity);  
      moveArm(direction);
      delayMicroseconds(stepDelay);  
   }
    ////////////// 
}
void  testMoveArmMM(SerialCommands *sender){
    int direction;
    float distance  = 10;
    Serial.println("Which direction would you like to go? (up = 1, down = 0)");
    direction = getIntegerFromUser();
    Serial.println("How many mm to travel?");
    distance = getFloatFromUser();
    moveArmMillimeters(distance, direction);
    Serial.println("Done moving arm.");}
void  testSpinBaseTwice(SerialCommands *sender){
    moveArmToMM(MIDDLE);
    closeHand();
    spinBaseTwice(true);
    openHand();
}


void  printOutMoveArmDebugMessage(int destination){
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

}
void moveArmToMM(int destination) {
  if(!gripperFunctional){
    Serial.println("Gripper not functional. Can't perform move.");
    return;
  }  
  printOutMoveArmDebugMessage(destination);
  int stepDelay = getDelay(moveArmSpeed);

  switch (destination) { 
    case BOTTOM:
      if(armLocation == BOTTOM){
        break;
      }
      if(armLocation == MIDDLE){
        moveArmMillimeters(middleHeight - bottomGripHeight, DOWN);
        armLocation = BOTTOM;
        break; 
      }
      if(armLocation == DROPOFF_HEIGHT){
        moveArmMillimeters((dropoffHeight - bottomGripHeight), DOWN);
        armLocation = BOTTOM;
        break; 
      }
      if(armLocation == TOP){
        moveArmMillimeters((topOfRotationHeight - bottomGripHeight), DOWN);
        armLocation = BOTTOM;
        break; 
      }
      if(armLocation == UNKNOWN){
        openHand();
        while(digitalRead(BOTTOM) == 1){
          moveArm(DOWN);
          delayMicroseconds(stepDelay);
        }
        moveArmMillimeters( (bottomGripHeight - bottomEndstopHeight) , UP);  
        armLocation = BOTTOM;
        break; 
      }     
       
 
    case MIDDLE:
      if(armLocation == MIDDLE){
        break;
      }
      if(armLocation == BOTTOM){
        moveArmMillimeters((middleHeight - bottomGripHeight), UP);  
        armLocation = MIDDLE;
        break; 
      }
      if(armLocation == DROPOFF_HEIGHT){
        moveArmMillimeters((dropoffHeight - middleHeight), DOWN);    
        armLocation = MIDDLE;
        break; 
      }
      if(armLocation == TOP){
        moveArmMillimeters((topOfRotationHeight - middleHeight), DOWN);     
        armLocation = MIDDLE;
        break; 
      }
      if(armLocation == UNKNOWN){
        openHand();
        while(digitalRead(BOTTOM) == 1){
          moveArm(DOWN);
          delayMicroseconds(stepDelay);
        }
        moveArmMillimeters((middleHeight - bottomEndstopHeight), UP);  
        armLocation = MIDDLE;
        break; 
      }  

    case DROPOFF_HEIGHT:
      if(armLocation == DROPOFF_HEIGHT){
        break;
      }
      if(armLocation == BOTTOM){
        moveArmMillimeters((dropoffHeight - bottomGripHeight), UP);  
        armLocation = DROPOFF_HEIGHT;
        break; 
      }
      if(armLocation == MIDDLE){
        moveArmMillimeters((dropoffHeight - middleHeight), UP);    
        armLocation = DROPOFF_HEIGHT;
        break; 
      }
      if(armLocation == TOP){
        moveArmMillimeters((topOfRotationHeight - dropoffHeight), DOWN);     
        armLocation = DROPOFF_HEIGHT;
        break; 
      }
      if(armLocation == UNKNOWN){
        openHand();
        while(digitalRead(BOTTOM) == 1){
          moveArm(DOWN);
          delayMicroseconds(stepDelay);
        }
        moveArmMillimeters( (dropoffHeight - bottomEndstopHeight) , UP);  
        armLocation = DROPOFF_HEIGHT;
        break;  
      }  
    
  
    case TOP:
      if(armLocation == TOP){
        break;
      }
      if(armLocation == BOTTOM){
        moveArmMillimeters((topOfRotationHeight - bottomGripHeight), UP);
        armLocation = TOP;
        break;
      }  
      if(armLocation == MIDDLE){
        moveArmMillimeters((topOfRotationHeight -middleHeight), UP);
        armLocation = TOP;
        break;
      }  
      if(armLocation == DROPOFF_HEIGHT){
        moveArmMillimeters((topOfRotationHeight - dropoffHeight), UP);     
        armLocation = TOP;
        break;
      }  
      if(armLocation == UNKNOWN){
        openHand();
        while(digitalRead(BOTTOM) == 1){
          moveArm(DOWN);
          delayMicroseconds(stepDelay);
        }
        moveArmMillimeters( (topOfRotationHeight - bottomEndstopHeight), UP);  
        armLocation = TOP;
        break;
      }    
  }
}
void homeArmAndHandMM(){
  openHand();
  moveArmToMM(TOP); //go to the top endstop
  double distanceToMiddleFromTopEndstop = topEndstopHeight - middleHeight;
  moveArmMillimeters(distanceToMiddleFromTopEndstop, DOWN);
  armLocation = MIDDLE;
}




void testMoveArmToMM(SerialCommands *sender){
  int exit = 0;
  while(exit == 0){
    Serial.println("Which location would you like to go to?: 1 = top, 2 = dropoff, 3 = middle, 4 = bottom, 5 = close hand, 6 = open hand, 7 = exit");
    int command = getIntegerFromUser();    
    
    switch(command){
      case 1:
        moveArmToMM(TOP);
        break;
      case 2:
        moveArmToMM(DROPOFF_HEIGHT);
        break;
      case 3:
        moveArmToMM(MIDDLE);
        break;
      case 4:
        moveArmToMM(BOTTOM);
        break;
      case 5:
        closeHand();    
        break;
      case 6:
        openHand();   
        break;
      case 7:
        Serial.println("Exit command received. Leaving test function.");
        exit = 1;  
        break;
      default:
        exit = 1;
        Serial.println("Unrecognized entry. Exiting function.");
        return;
    }
  }
}
// last left off trying to implement the new movement function where we use moveArmToMM whenever possible. Lets see if this pans out.
// this must be finished for zen mode to be truly perfect (0 sound)
// current issue is that we need to make sure that when we go to top we're actually going up high enough to spin the cube.
void speeen(SerialCommands * sender) {
  int startTime = 0;
  openHand();
  while (!Serial.available()) {
    startTime = millis();
    flipCube();
    int elapsed = millis() - startTime;
    Serial.print("Time to flip cube = ");
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
    lastMove = randomMove; //FLAG - this does not actually prevent the previous move to be cancelled out due to the transformation that occurs to the cube when it is flipped ie topCW for move2
    //isn't opped by topCCW on move 1
  }

  //revert to hardcoded default speeds
  spinSpeed          = defaultSpinSpeed;
  moveArmSpeed       = defaultMoveArmSpeed;
  handOpenCloseSpeed = defaultHandOpenCloseSpeed;
  }
void scramble(SerialCommands * sender) {
  int lastMove = TOPCW;  
  int numScrambles = 50;
  float startTime = millis();
  for(int i = 0; i < numScrambles; i++){
    Serial.print("////////////////////////////// Now performing scramble number "); Serial.println( i + 1);
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
}

//initialize serialCommand functions
SerialCommand setGripDistance_("t", setGripDistance);
SerialCommand testDistanceToMiddleFromTop_("m", testDistanceToMiddleFromTop);
SerialCommand testDistanceToMiddleFromBottom_("m2", testDistanceToMiddleFromBottom);
SerialCommand setCubeError_("configCube", setCubeError);
SerialCommand setDistanceToMiddleFromCubeRelease_("d2", setDistanceToMiddleFromCubeRelease);
SerialCommand setZenSpeeds_("setZen", setZenSpeeds);
SerialCommand testMove_("testMove", testMove);
SerialCommand testCorrection_("testCorr", testCorrection);
SerialCommand toggleSteppers_("tog", toggleSteppers);
SerialCommand setbetweenActionsDelay_("setD", setbetweenActionsDelay);
SerialCommand setScrambleAndSolveSpeed_("setSpeed", setScrambleAndSolveSpeed);
SerialCommand scanLightCurrent_("scan", scanLightCurrent);
SerialCommand fixCubeError_("fix", fixCubeError);
SerialCommand readGripperCurrent_("gg", readGripCurrent);
SerialCommand testNewCloseHand_("ch", testNewCloseHand);
SerialCommand measureDistanceTopToBottom_("t2b", measureDistanceTopToBottom);
SerialCommand testMoveArmMM_("tma",testMoveArmMM);
SerialCommand testMoveArmToMM_("tmaMM", testMoveArmToMM);
SerialCommand testSpinBaseTwice_("tsb2", testSpinBaseTwice);

SerialCommand speeen_("speeen", speeen);
SerialCommand zenMode_("zen", zenMode); // zen mode will slowly and infinitely scramble until serial input is received (will finish current move)
SerialCommand scramble_("scramble", scramble); 

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
serial_commands_.AddCommand(&setbetweenActionsDelay_);
serial_commands_.AddCommand(&setScrambleAndSolveSpeed_);
serial_commands_.AddCommand(&scanLightCurrent_);
serial_commands_.AddCommand(&fixCubeError_);
serial_commands_.AddCommand(&readGripperCurrent_);
serial_commands_.AddCommand(&testNewCloseHand_);
serial_commands_.AddCommand(&measureDistanceTopToBottom_);
serial_commands_.AddCommand(&testMoveArmMM_);
serial_commands_.AddCommand(&testMoveArmToMM_);
serial_commands_.AddCommand(&testSpinBaseTwice_);

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

if (! lightRingINA.begin()) {
  Serial.println("Failed to find Light Ring Curret sensor");
}

if (! gripperINA.begin()) {
  Serial.println("Failed to find Gripper Curret sensor");
}


Serial.println("Ready!");
// homeArmAndHand();
homeArmAndHandMM();
// moveArmToMM(MIDDLE);
// homeBase();
// homeBase2();
// homeBase3();
// homeBase4();
// homeBase5();
}

void loop() {
  int raise = digitalRead(raiseArmButton);
  int lower = digitalRead(lowerArmButton);
  int open  = digitalRead(openHandButton);
  int close = digitalRead(closeHandButton);
  int spin  = digitalRead(spinBaseButton);

  if (raise == 0) {
    moveArm(UP);    
    armLocation = UNKNOWN;
  }

  if (lower == 0) {
    moveArm(DOWN);
    armLocation = UNKNOWN;
  }

  if (open == 0) {
    articulateHand(OPEN);
    handState = UNKNOWN;
  }

  if (close == 0) {
    articulateHand(CLOSE);
    handState = UNKNOWN;
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
          SerialBT.write(ACK);
          break;

        case 'z':
          zenMode(NULL);
          SerialBT.write(ACK);
          break;

        case 'r':
          speeen(NULL);
          SerialBT.write(ACK);
          break;

        case 't':
          toggleSteppers(NULL);
          SerialBT.write(ACK);
          break;    

        // Cubing notation moves
        case 'b':
          moveArmToMM(MIDDLE);
          closeHand();
          spinBase(cw, true);
          openHand();
          SerialBT.write(ACK);
          break;

        case 'B':
          moveArmToMM(MIDDLE);
          closeHand();
          spinBase(ccw, true);
          openHand();
          SerialBT.write(ACK);
          break;

        case 'p':
          moveArmToMM(MIDDLE);
          closeHand();
          spinBaseTwice(true);
          openHand();
          SerialBT.write(ACK);
          break;

        case 'y':
          spinBase(ccw, false);
          SerialBT.write(ACK);
          break;

        case 'Y':
          spinBase(cw, false);
          SerialBT.write(ACK);
          break;

        case 'P':
          spinBaseTwice(false);
          break;

        case 'X':
          flipCube();
          SerialBT.write(ACK);
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


