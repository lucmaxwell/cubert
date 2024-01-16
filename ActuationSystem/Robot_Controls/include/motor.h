//******************************************************************************************************
//
//	@file 		motor.h
//	@author 	Bruno Di Gaetano, Matthew Mora
//	@created	Jan 12, 2024
//	@brief		Contains controls for activating motors
//
//******************************************************************************************************

#ifndef _MOTOR
#define _MOTOR

#include <Arduino.h>
#include <stdint.h>

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
// #define OPEN      1
// #define CLOSE     0
#define MIDDLE  420
#define BOTTOM   endstop_arm_lowerLimit_pin
#define TOP      endstop_arm_upperLimit_pin


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

// #define HandStates      uint8_t
// #define ArmLocations    uint8_t

// #define UNKNOWN     0

// #define OPEN        1
// #define CLOSED      2

// #define TOP         1
// #define MIDDLE      2
// #define BOTTOM      3

namespace HandState{
    enum HandStates{
        UNKNOWN,
        OPEN,
        CLOSED,
    };
}

namespace ArmLocation{
    enum ArmLocations{
        UNKNOWN,
        TOP_ARM,
        MIDDLE_ARM,
        BOTTOM_ARM,
    };
}

HandState::HandStates handState         = HandState::UNKNOWN;
ArmLocation::ArmLocations armLocation   = ArmLocation::UNKNOWN;

//manual button pin assignments
#define raiseArmButton  33
#define lowerArmButton  26
#define openHandButton  25
#define closeHandButton 32
#define spinBaseButton  27

void moveArm(int direction);
void homeArmAndHand();
void moveArmTo(int destination);
void closeHand();
void openHand();
int getDelay(int v);
int getIntegerFromUser();

#endif //_MOTOR