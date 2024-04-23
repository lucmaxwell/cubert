#include "../include/motor.h"

// defines

// pin assignements
#define motors_en_pin              26      // LOW: Driver enabled. HIGH: Driver disabled

#define motors_base_step_pin       27      // Step on rising edge
#define motors_base_dir_pin        17

#define motors_arm_left_dir_pin     6
#define motors_arm_left_step_pin    5

#define motors_arm_right_dir_pin   19
#define motors_arm_right_step_pin  13

#define endstop_arm_openLimit_pin  16
#define endstop_arm_upperLimit_pin 20
#define endstop_arm_lowerLimit_pin 21

///////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_SPEED 3.3        // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
#define MIN_SPEED 0.000001   // DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
///////////////////////////////////////////////////////////////////////////////////////////////////////////

// function prototypes

void initMotor();

int getDelay(int velocity);
void step(MotorType motor, Direction direction);


// function methods

void initMotor()
{
    WiringPiSetup();
    pinMode
}

int getDelay(int velocity)
{
    velocity = min(velocity, 200);
    double x = MIN_SPEED + velocity * (MAX_SPEED - MIN_SPEED) / 100;
    double delayDuration = pow(0.0003 * x, -1) / 10;

    return round(delayDuration);
}

void step(MotorType motor, Direction direction)
{
    // function variables
    int motor_dir_pin;
    int motor_step_pin;

    // determine motor to step
    switch (motor)
    {
        case BASE:
            motor_dir_pin = motors_base_dir_pin;
            motor_step_pin = motors_base_step_pin;
            break;
        case LEFT:
            motor_dir_pin = motors_arm_left_dir_pin;
            motor_step_pin = motors_arm_left_step_pin;
            break;
        case RIGHT:
            motor_dir_pin = motors_arm_right_dir_pin;
            motor_step_pin = motors_arm_right_step_pin;
            break;
        default:
            break;
    }

    // set motor direction
    digitalWrite(motor_dir_pin, int(direction));
    
    // step motor
    digitalWrite(motor_step_pin, !digitalRead(motor_step_pin)); 
}

// int 

// for (int i = 0; i < numSteps; i++) {        
//        if(i < point1){
//         velocity = moveArmSpeed * double(i)/point1;        
//        }
//        else if(i >= point1 && i <= point2){
//         velocity = moveArmSpeed;
//       }else{
//         velocity = moveArmSpeed * (numSteps - double(i)) / point1;
//       }
//       velocity  = max(velocity, minimumArmSpeed);
//       stepDelay = getDelay(velocity);  
//       moveArm(direction);
//       delayMicroseconds(stepDelay);  
//    }