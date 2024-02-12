import Motor
import time
import signal
import sys
import RPi.GPIO as GPIO
import random
from CurrentSensor import *
from enum import IntEnum

class CubertNotation(IntEnum):
    BOTTOM_CW   = 0
    BOTTOM_CCW  = 1
    TOP_CW      = 2
    TOP_CCW     = 3
    FRONT_CW    = 4
    FRONT_CCW   = 5
    BACK_CW     = 6
    BACK_CCW    = 7
    LEFT_CW     = 8
    LEFT_CCW    = 9
    RIGHT_CW    = 10
    RIGHT_CCW   = 11

def sigint_handler(sig, frame):
    del actions
    del motor

    GPIO.cleanup()
    sys.exit(0)

class CubertActions:

    _defaul_move_speed = 10

    def __init__(self, motor:Motor.CubertMotor, calibrate_distance=False, default_move_speed=10):
        self.motor = motor
        self._defaul_move_speed = default_move_speed

        motor.enable()

        self.motor.home(calibrate_distance)

    def preformMove(self, move:CubertNotation, rotation:Motor.BaseRotation, move_speed=_defaul_move_speed):

        if move == CubertNotation.BACK_CCW:
            self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.BACK_CW:
            self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.BOTTOM_CCW:
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.BOTTOM_CW:
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.FRONT_CCW:
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.BOTTOM_CW:
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.LEFT_CCW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)
            
        elif move == CubertNotation.LEFT_CW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.RIGHT_CCW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)
            
        elif move == CubertNotation.RIGHT_CW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.TOP_CCW:
            self.flip()
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.TOP_CW:
            self.flip()
            self.flip()
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)



    def flip(self, move_speed=10):
        self.motor.moveGripperToPos(Motor.GripperPosition.BOTTOM)
        self.motor.closeHand()
        self.motor.moveGripperToPos(Motor.GripperPosition.TOP)
        self.motor.openHand()

    def rotateFace(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_defaul_move_speed):
        self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE)
        self.motor.closeHand()
        self.motor.spinBase(rotation, direction, degrees_to_correct=15)
        self.motor.openHand()

    def rotateCube(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_defaul_move_speed):
        self.motor.spinBase(rotation, direction)

    def zen(self, move_speed=10):
        while True:
            move = random.randint(0,11)

            if random.randint(0,1):
                rotation = Motor.BaseRotation.QUARTER
            else:
                rotation = Motor.BaseRotation.HALF

            self.preformMove(move, rotation, move_speed)
  
            # if move == 2:
            #     self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            # elif move == 3:
            #     self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            # elif move == 4:
            #     self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            # elif move == 5:
            #     self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW, move_speed)
            # elif move == 6:
            #     self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            # elif move == 7:
            #     self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            # elif move == 8:
            #     self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            # elif move == 9:
            #     self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CW, move_speed)
            # else:
            #     self.flip()

    

if __name__ == '__main__':
    motor_en_pin = 26
    motor_step_pin = [27, 6, 19]
    motor_dir_pin = [17, 5, 13]

    # End stop for arm
    end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
    end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
    end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

    signal.signal(signal.SIGINT, sigint_handler)

    current_sensor = CubertCurrentSensor()

    # initialize motor
    motor = Motor.CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin, current_sensor)

    actions = CubertActions(motor,)

    time.sleep(2)

    actions.flip()
    time.sleep(1)
    actions.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)
    time.sleep(1)
    actions.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW)
    time.sleep(1)
    actions.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
    time.sleep(1)
    actions.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CCW)

    actions.zen()


    del actions
    del motor

    GPIO.cleanup()

