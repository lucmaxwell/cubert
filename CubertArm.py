from enum import Enum

import RPi.GPIO as GPIO
import time

from Button import CubertButton
from Motor import CubertMotor, MotorSpin


# Parameters
NUM_STEPS_CALIBRATION = 20
NUM_STEPS_FROM_TOP_TO_MIDDLE = 1200
NUM_STEPS_FROM_TOP_TO_DROP_OFF = 600
NUM_STEPS_FROM_DROP_OFF_TO_MIDDLE = 700
GRIP_STRENGTH = 350


class ArmDirection(Enum):
    UP = 0
    DOWN = 1


class HandAction(Enum):
    OPEN = 0
    CLOSE = 1


class ArmPosition(Enum):
    BOTTOM = 0
    MIDDLE = 1
    TOP = 2
    DROP_OFF = 3


class CubertArm:
    def __init__(self, enable_pin, left_motor_pin_list, right_motor_pin_list, end_stop_arm_pin_list):
        # Initialize the motors
        self.left_motor = CubertMotor(enable_pin, left_motor_pin_list)
        self.right_motor = CubertMotor(enable_pin, right_motor_pin_list)

        # Initialize the end stop
        self.end_stop_arm_upper_limit = CubertButton(end_stop_arm_pin_list[0], GPIO.LOW)
        self.end_stop_arm_lower_limit = CubertButton(end_stop_arm_pin_list[1], GPIO.LOW)
        self.end_stop_hand_open_limit = CubertButton(end_stop_arm_pin_list[2], GPIO.LOW)

        # # Ensure that the hand is open
        # self.open_hand(60)
        #
        # # Arm calibration
        # self.arm_calibration = 0
        # while not self.end_stop_arm_lower_limit.pressed():
        #     self.move_arm_to_bottom()
        #
        #
        #
        #
        # self.arm_position = ArmPosition.MIDDLE
        # self.move_arm_to_bottom(60)
        # while not self.end_stop_arm_upper_limit.pressed():
        #     self.move_arm(ArmDirection.UP, 60)
        #     self.arm_calibration += 1
        #
        # # Move the arm to default middle to position to indicate calibration completed
        # self.move_arm_to(ArmPosition.MIDDLE, 60)

    def calibrate(self, arm_position, move_speed):
        move_direction = ArmDirection.DOWN
        if arm_position == ArmPosition.BOTTOM:
            self.move_arm(ArmDirection.UP, move_speed)

        # Calibrate
        if arm_position == ArmPosition.TOP or arm_position == ArmPosition.BOTTOM:
            for _ in range(NUM_STEPS_CALIBRATION):
                self.move_arm(move_direction, move_speed)

    def move_arm(self, arm_direction, move_speed):
        # End stop is used to determine if the motor can move the arm
        if arm_direction == ArmDirection.UP and not self.end_stop_arm_upper_limit.pressed():
            # Enable the motors
            self.left_motor.enable()
            self.right_motor.enable()

            # Move to the correct direction
            self.left_motor.step(1, MotorSpin.CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)
        elif arm_direction == ArmDirection.DOWN and not self.end_stop_arm_lower_limit.pressed():
            # Enable the motors
            self.left_motor.enable()
            self.right_motor.enable()

            # Move to the correct direction
            self.left_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.CLOCKWISE, move_speed)

    def move_hand(self, open_close_flag, move_speed):
        if open_close_flag == HandAction.OPEN and not self.end_stop_hand_open_limit.pressed():
            self.left_motor.step(1, MotorSpin.CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.CLOCKWISE, move_speed)
        elif open_close_flag == HandAction.CLOSE:
            self.left_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)

    def move_arm_to_top(self, move_speed):
        self.arm_position = ArmPosition.TOP
        while not self.end_stop_arm_upper_limit.pressed():
            self.move_arm(ArmDirection.UP, move_speed)

    def move_arm_to_bottom(self, move_speed):
        self.arm_position = ArmPosition.BOTTOM
        while not self.end_stop_arm_lower_limit.pressed():
            self.move_arm(ArmDirection.DOWN, move_speed)

    def move_arm_to(self, position, move_speed):
        if position == ArmPosition.TOP:
            self.move_arm_to_top(move_speed)
        elif position == ArmPosition.BOTTOM:
            self.move_arm_to_bottom(move_speed)
        elif position == ArmPosition.MIDDLE:
            # From top
            if self.arm_position == ArmPosition.TOP:
                for _ in range(NUM_STEPS_FROM_TOP_TO_MIDDLE):
                    self.move_arm(ArmDirection.DOWN, move_speed)

            # From drop off
            elif self.arm_position == ArmPosition.DROP_OFF:
                for _ in range(NUM_STEPS_FROM_DROP_OFF_TO_MIDDLE):
                    self.move_arm(ArmDirection.DOWN, move_speed)

        elif position == ArmPosition.DROP_OFF:
            if self.arm_position == ArmPosition.TOP:
                for _ in range(NUM_STEPS_FROM_TOP_TO_DROP_OFF):
                    self.move_arm(ArmDirection.DOWN, move_speed)

        # Update the position of the arm
        self.arm_position = position

    def close_hand(self, move_speed):
        for _ in range(GRIP_STRENGTH):
            self.move_hand(HandAction.CLOSE, move_speed)

    def open_hand(self, move_speed):
        while not self.end_stop_hand_open_limit.pressed():
            self.move_hand(HandAction.OPEN, move_speed)
