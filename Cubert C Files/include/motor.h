#include <wiringPi.h>

#ifndef _MOTOR

#define _MOTOR

enum Direction{
    CCW = 0,
    CW  = 1,
};

enum MotorType{
    BASE    = 0,
    LEFT    = 1,
    RIGHT   = 2,
};

#endif //_MOTOR