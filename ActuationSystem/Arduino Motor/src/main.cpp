#include <Arduino.h>
#include <stdio.h>
#include <SerialCommands.h>
#define MAX_SPEED 3     // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK THE STEPPERS. (2.1)
#define MIN_SPEED 0.000001  // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK THE STEPPERS.

#define EN_PIN    14                  // LOW: Driver enabled. HIGH: Driver disabled
#define DIR_PIN   32
#define STEP_PIN  33              // Step on rising edge
#define cw 0
#define ccw 1

char serial_command_buffer_[32];

// function prototypes
void cmd_unrecognized(SerialCommands*, const char*);
void toggleStepper(SerialCommands*);
void callTesterFunction(SerialCommands*);
void clockwise(SerialCommands*);
void counterClockwise(SerialCommands*);
void displayMe(SerialCommands*);
void fast(SerialCommands*);
int getDelay(int);
void spinBase(int, float, int);
void spinBaseAccelerating(int, float, int);

SerialCommands serial_commands_(&Serial, serial_command_buffer_, sizeof(serial_command_buffer_), "\r\n", " ");

SerialCommand clockwise_("r", clockwise);
SerialCommand counterClockwise_("l", counterClockwise);
SerialCommand displayMe_("d", displayMe);
SerialCommand fast_("f", fast);
SerialCommand toggleStepper_("t", toggleStepper);
SerialCommand callTesterFunction_("waltz", callTesterFunction);

void setup() {
  Serial.begin(115200);
  serial_commands_.SetDefaultHandler(cmd_unrecognized);
  serial_commands_.AddCommand(&clockwise_);
  serial_commands_.AddCommand(&counterClockwise_);
  serial_commands_.AddCommand(&displayMe_);
  serial_commands_.AddCommand(&fast_);
  serial_commands_.AddCommand(&toggleStepper_);
  serial_commands_.AddCommand(&callTesterFunction_);

  pinMode(EN_PIN, OUTPUT);
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW); // enable stepper
  Serial.println("Ready!");
}

void loop() {                                     // set the direction  
  serial_commands_.ReadSerial();
}

void cmd_unrecognized(SerialCommands* sender, const char* cmd){
  sender->GetSerial()->println("Unrecognized command");
}

void toggleStepper(SerialCommands* sender){   
  int state = digitalRead(EN_PIN);
  if(state){
      digitalWrite(EN_PIN, LOW);
      sender->GetSerial()->println("Enabling rotation stepper.");
      }
  else{
      digitalWrite(EN_PIN, HIGH);
      sender->GetSerial()->println("Disabling rotation stepper.");
      }        
}

void callTesterFunction(SerialCommands* sender){  
  // Serial.println("Now testing the base accelerator output.");
  long choose;
  long randDelay;
  int slow = 25; int fast = 50;
  while(!Serial.available()){ 
    choose = random(0,15);
    if(choose < 5){
        spinBaseAccelerating(cw, 90, slow);
        delay(50);
        spinBaseAccelerating(ccw, 90, slow);
        delay(50);
        spinBaseAccelerating(ccw, 90, slow);
    }
    if(choose<10 && choose>5){
        spinBaseAccelerating(ccw, 90, slow);
        delay(50);
        spinBaseAccelerating(cw, 90, slow);
        delay(50);
        spinBaseAccelerating(cw, 90, slow);
    }
    else{
      spinBaseAccelerating(ccw, 180, fast);
    }
    delay(50);
  }
      
}

void clockwise(SerialCommands* sender){   
  spinBaseAccelerating(cw, 96, 35); 
  sender->GetSerial()->println("Turning clockwise."); 
  delay(500);
  spinBaseAccelerating(ccw, 6, 35); 
}

void counterClockwise(SerialCommands* sender){   
  spinBaseAccelerating(ccw, 90, 35);
  sender->GetSerial()->println("Turning counter clockwise.");
}

void displayMe(SerialCommands* sender){
  sender->GetSerial()->println("Displaying cube."); 
  while(!Serial.available()){
   spinBase(cw,360,1); 
   sender->GetSerial()->println("Keep rollin rollin rollin rollin...");   
  }
}

void fast(SerialCommands* sender){
  int counter = 1;
  sender->GetSerial()->println("Sicko mode engaged..."); 
  while(!Serial.available()){
  spinBase(cw,360,100); 
    Serial.println("Revolutions completed: " + String(counter));   
    sender->GetSerial()->println("Keep rollin rollin rollin rollin..."); 
    counter++;     
  }
  sender->GetSerial()->print("Exiting sicko mode. We completed "); 
  sender->GetSerial()->print(counter); 
  sender->GetSerial()->println(" revolutions."); 
  spinBase(cw,366,100); 
  delay(500);
  spinBase(ccw, 6, 35);  
}

int getDelay(int v){    
  v = min(v, 100);
  double x = MIN_SPEED + v*(MAX_SPEED - MIN_SPEED)/100;
  double delayDuration = pow(0.0003*x,-1)/10; 
  return round(delayDuration);  
}

void spinBase(int my_direction, float deg, int v){ 
  int stepDelay = getDelay(v);  
  digitalWrite(DIR_PIN, my_direction);                                    // set the direction  
  float steps = 3200 * deg/360.0;  
  for (int i = 0 ;i < steps; i++) {
    digitalWrite(STEP_PIN, !digitalRead(STEP_PIN)); // Step
    delayMicroseconds(stepDelay);
  }
}
  
void spinBaseAccelerating(int my_direction, float deg, int v){
  //creates an array of delays between step instructions such that the stepper smoothly accelerates (sinusoidally from 0 -> pi/2).

  //IT IS VERY IMPORTANT THAT WHEN THIS FUNCTION IS TRANSLATED TO MICROPYTHON THAT WE CHECK THE DELAY VALUES BEFORE WE SEND THEM 
  //TO THE STEPPER CONTROLLER. OR ELSE.
  
  digitalWrite(DIR_PIN, my_direction); // set rotational direction
  double piOverTwo = 1.5708;
  v = min(v, 100);// just in case someone accidentally does something... interesting
  float steps = 2* 3200 * deg/360.0;// yeah these two lines are probably poo poo caca
  int numSteps = round(steps);// yeah these two lines are probably poo poo caca 
  float delaysArray[numSteps];
  int halfLength = floor(numSteps/2);
  int counter = 0;
  double x;

  for(int i = 1; i <= halfLength; i++){ // fill the first half of the array
    x = (MIN_SPEED + v*(MAX_SPEED - MIN_SPEED)/100) * sin(piOverTwo*(double(i)/double(halfLength)));// x output is supposed to be in rotations/second
    // Serial.print("X = ");
    // Serial.println(x);      
    delaysArray[i-1] = pow(x,-1)*300;// convert x from rotations per second to microsecond delay 
  }

  if(numSteps % 2){// if we have an odd number of steps we need to take, then fill in the bald spot with a copy of the lowest delay
    Serial.println("Fizzbuzz! We received an array of odd length!");
    delaysArray[halfLength] = delaysArray[halfLength-1];      
  }

  for(int i = numSteps; i > halfLength; i--){//fill in the second half of the array, backwards
      delaysArray[i-1] = delaysArray[counter];
      counter++;
  }

  // for (int i = 0; i < numSteps; i++){ // debugging garbage
  //   Serial.println(delaysArray[i]); 
  // }

  for (int i = 0 ;i < numSteps; i++) { // oh shoot here it comes, we sinusoidal as heck boooiii
    digitalWrite(STEP_PIN, !digitalRead(STEP_PIN)); 
    delayMicroseconds(floor(delaysArray[i]));
  }  
}