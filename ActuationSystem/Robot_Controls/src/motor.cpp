//******************************************************************************************************
//
//	@file 		motor.cpp
//	@authors 	Bruno Di Gaetano, Matthew Mora
//	@created	Jan 12, 2024
//	@brief		Contains controls for activating motors
//
//******************************************************************************************************

#include "motor.h"

void articulateHand(HandState::HandStates);

void homeArmAndHand() {
//   String desiredLocation = "";
//   Serial.print("Homing hand to ");
//     switch (homePosition) {
//         case TOP:
//             // desiredLocation = "top";
//             break;
//         case BOTTOM:
//         // desiredLocation = "bottom";
//         break;
//         case MIDDLE:
//         // desiredLocation = "middle";
//         break;
//     }
// //   Serial.println(desiredLocation);
    openHand();
    moveArmTo(homePosition);
}

void moveArmTo(int destination) {
//   String desiredLocation = "";
//   Serial.print("Moving arm from ");
//   Serial.print(armLocation);
//   Serial.print(" to ");
//   switch (destination) {
//     case TOP:
//       desiredLocation = "top";
//       break;
//     case BOTTOM:
//       desiredLocation = "bottom";
//       break;
//     case MIDDLE:
//     desiredLocation = "middle";
//     break;
//   }
//   Serial.println(desiredLocation);
  switch (destination) {
    /////////////////////////////////////////////////////////
    case BOTTOM:
      while (1) {
        if (digitalRead(destination) == 0)
          break;
        moveArm(DOWN);
      }
      armLocation = ArmLocation::BOTTOM_ARM;
      break;
    /////////////////////////////////////////////////////////
    case TOP:
      while(1){
        if(digitalRead(destination) == 0)
          break;
      moveArm(UP);
      }     
      armLocation = ArmLocation::BOTTOM_ARM;
      break;
    ///MIDDLE/////////////////////////////////////////////////////
    case MIDDLE:
      if(armLocation == ArmLocation::BOTTOM_ARM){
        break;
      }
      if(armLocation == ArmLocation::TOP_ARM){          ;
          for(int i = 0; i < numStepsFromTopToMiddle; i++){
            moveArm(DOWN);
            break;
          }
      }
      else if (armLocation == ArmLocation::BOTTOM_ARM){
        //   Serial.println(" from the bottom.");
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
      armLocation = ArmLocation::MIDDLE_ARM;
    break;    
  }
    // Serial.println("Arrived at destination.");
    delay(interActionDelay);
}

void closeHand() {
  for (int i = 0; i < gripStrength; i++) {
    articulateHand(HandState::CLOSED);
  }
  delay(interActionDelay);
  handState = HandState::OPEN; 
}

void openHand() {
  Serial.println("Opening hand");
  while (1) {
    if (digitalRead(endstop_arm_openLimit_pin) == 0)
      return;
    articulateHand(HandState::OPEN);
  }
  handState = HandState::CLOSED;
  delay(interActionDelay); 
}

void spinBase(int my_direction, float deg, int v) {
  int stepDelay = getDelay(v);
  if (armLocation != ArmLocation::TOP_ARM && armLocation != ArmLocation::MIDDLE_ARM) {
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

void dropCubeToBase() {
  for (int i = 0; i < cubeDropDistance; i++) {
    moveArm(DOWN);
  }
  delay(interActionDelay);
  armLocation = ArmLocation::UNKNOWN; // lol imagine forgetting to put this here xD
}

void articulateHand(HandState::HandStates direction) {
  handState = HandState::UNKNOWN;
  //set the direction that the arm motors spin
  if (direction == HandState::OPEN) {
    // Serial.println("Opening");
    digitalWrite(motors_arm_left_dir_pin, cw);
    digitalWrite(motors_arm_right_dir_pin, cw);
  } else {
    // Serial.println("Closing");
    digitalWrite(motors_arm_left_dir_pin, ccw);
    digitalWrite(motors_arm_right_dir_pin, ccw);
  }

  if (digitalRead(endstop_arm_openLimit_pin) == 1 || direction == HandState::CLOSED) {
    digitalWrite(motors_arm_left_step_pin, !digitalRead(motors_arm_left_step_pin));    // Step left motor
    digitalWrite(motors_arm_right_step_pin, !digitalRead(motors_arm_right_step_pin));  // Step right motor
  }
  int stepDelay = getDelay(handOpenCloseSpeed);
  delayMicroseconds(stepDelay);
}

void clockwise(){//SerialCommands * sender) {
  spinBase(cw, numDegreesToRotate, spinSpeed);
}

void counterClockwise(){//SerialCommands * sender) {
  spinBase(ccw, numDegreesToRotate, spinSpeed);
//   sender->GetSerial()->println("Turning counter clockwise.");
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
  armLocation = ArmLocation::MIDDLE_ARM;
  handState = HandState::CLOSED;
  delay(interActionDelay);
}

void scramble(){//SerialCommands * sender) {
  int numScrambles = 0;
  if(armLocation == ArmLocation::UNKNOWN)
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

void testGripDistance(){//SerialCommands * sender) {
  while (digitalRead(endstop_arm_openLimit_pin) == 1) {
    articulateHand(HandState::OPEN);
  }
//   Serial.println("Please enter how many steps to close the gripper for:");
  int numSteps = getIntegerFromUser();
//   Serial.print("Now closing gripper for ");
//   Serial.print(numSteps, DEC);
//   Serial.println(" steps.");
  for (int i = 0; i < numSteps; i++) {
    articulateHand(HandState::CLOSED);
  }
}