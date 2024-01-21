from enum import Enum

import RPi.GPIO as GPIO
import time

from Button import CubertButton
from Motor import CubertMotor, MotorSpin


class ArmDirection(Enum):
    UP = 0
    DOWN = 1


class CubertArm:
    def __init__(self, enable_pin, left_motor_pin_list, right_motor_pin_list, endstop_arm_upper_pin, endstop_arm_lower_pin):
        # Initialize the motors
        self.left_motor = CubertMotor(enable_pin, left_motor_pin_list)
        self.right_motor = CubertMotor(enable_pin, right_motor_pin_list)

        # Initialize the end stop
        self.end_stop_arm_upper_limit = CubertButton(endstop_arm_upper_pin, GPIO.LOW)
        self.end_stop_arm_lower_limit = CubertButton(endstop_arm_lower_pin, GPIO.LOW)

    def move(self, arm_direction):
        # End stop is used to determine if the motor can move the arm
        if arm_direction == ArmDirection.UP and not self.end_stop_arm_upper_limit.pressed():
            # Enable the motors
            self.left_motor.enable()
            self.right_motor.enable()

            # Move to the correct direction
            self.left_motor.step(1, MotorSpin.CLOCKWISE)
            self.right_motor.step(1, MotorSpin.COUNTER_CLOCKWISE)
        elif arm_direction == ArmDirection.DOWN and not self.end_stop_arm_lower_limit.pressed():
            # Enable the motors
            self.left_motor.enable()
            self.right_motor.enable()

            # Move to the correct direction
            self.left_motor.step(1, MotorSpin.COUNTER_CLOCKWISE)
            self.right_motor.step(1, MotorSpin.CLOCKWISE)
