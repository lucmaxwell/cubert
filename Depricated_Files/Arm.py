from enum import Enum

import RPi.GPIO as GPIO
import time

from Button import CubertButton
from Motor import CubertMotor, MotorSpin


# Parameters
NUM_STEPS_CALIBRATION = 20
ARM_MIDDLE_POSITION_RATIO = 0.5
ARM_DROP_OFF_POSITION_RATIO = 0.8
# NUM_STEPS_FROM_TOP_TO_MIDDLE = 1200
# NUM_STEPS_FROM_TOP_TO_DROP_OFF = 600
# NUM_STEPS_FROM_DROP_OFF_TO_MIDDLE = 700
GRIP_STRENGTH = 350


class ArmDirection(Enum):
    UP = 0
    DOWN = 1


class HandAction(Enum):
    OPEN = 0
    CLOSE = 1


class ArmPosition(Enum):
    BOTTOM_ENDSTOP = 0
    MIDDLE = 1
    TOP_ENDSTOP = 2
    DROP_OFF = 3


class Arm:
    def __init__(self, enable_pin, left_motor_pin_list, right_motor_pin_list, end_stop_arm_pin_list):
        # Initialize the motors
        self.left_motor = CubertMotor(enable_pin, left_motor_pin_list)
        self.right_motor = CubertMotor(enable_pin, right_motor_pin_list)

        # Initialize the end stop
        self.end_stop_arm_upper_limit = CubertButton(end_stop_arm_pin_list[0], GPIO.LOW)
        self.end_stop_arm_lower_limit = CubertButton(end_stop_arm_pin_list[1], GPIO.LOW)
        self.end_stop_hand_open_limit = CubertButton(end_stop_arm_pin_list[2], GPIO.LOW)

        # Ensure that the hand is open
        self.open_hand(60)
        
        # First time calibration
        # Arm calibration move to the bottom and hit the end stop
        self.move_arm_to_bottom(60)
        
        # Count the number of steps to reach the top end stop
        self.arm_total_steps = 0
        while not self.end_stop_arm_upper_limit.pressed():
            # Move the arm by one step
            self.move_arm(ArmDirection.UP, 60)

            # Update the count
            self.arm_total_steps += 1

        # Buffer step calibrate
        self.arm_total_steps -= NUM_STEPS_CALIBRATION * 2

        # Recalibrate the arm position
        self.current_arm_position = 0
        self.recalibrate(60)
        
        # Move the arm to default middle to position to indicate calibration completed
        self.move_arm_to(ArmPosition.MIDDLE, 60)

    def recalibrate(self, move_speed):
        # Recalibrate based on the end stop that was hit
        if self.end_stop_arm_upper_limit.pressed or self.end_stop_arm_lower_limit.pressed:
            self.current_arm_position = 0
            move_direction = ArmDirection.UP
            if self.end_stop_arm_lower_limit.pressed():
                self.current_arm_position = self.arm_total_steps
                move_direction = ArmDirection.DOWN
            
            # Step adjustment
            for _ in range(NUM_STEPS_CALIBRATION):
                self.move_arm(move_direction, move_speed)

    def enable_arm_motors(self):
        self.left_motor.enable()
        self.right_motor.enable()

    def disable_arm_motors(self):
        self.left_motor.disable()
        self.right_motor.disable()


    def move_arm(self, arm_direction, move_speed):
        # Enable the motors
        self.enable_arm_motors()

        # End stop is used to determine if the motor can move the arm
        if arm_direction == ArmDirection.UP and not self.end_stop_arm_upper_limit.pressed():
            # Move to the correct direction
            self.left_motor.step(1, MotorSpin.CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)

            # Disable the motor
            self.left_motor.disable()
            self.right_motor.disable()

        elif arm_direction == ArmDirection.DOWN and not self.end_stop_arm_lower_limit.pressed():
            # Move to the correct direction
            self.left_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.CLOCKWISE, move_speed)

        # Disable the motors
        self.disable_arm_motors()

    def move_hand(self, open_close_flag, move_speed):
        # Enable the motors
        self.enable_arm_motors()

        # Move the hand with given flag direction
        if open_close_flag == HandAction.OPEN and not self.end_stop_hand_open_limit.pressed():
            self.left_motor.step(1, MotorSpin.CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.CLOCKWISE, move_speed)
        elif open_close_flag == HandAction.CLOSE:
            self.left_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)
            self.right_motor.step(1, MotorSpin.COUNTER_CLOCKWISE, move_speed)

        # Disable the motors
        self.disable_arm_motors()

    def move_arm_to_top(self, move_speed):
        self.current_arm_position = self.arm_total_steps
        while not self.end_stop_arm_upper_limit.pressed():
            self.move_arm(ArmDirection.UP, move_speed)

    def move_arm_to_bottom(self, move_speed):
        self.current_arm_position = 0
        while not self.end_stop_arm_lower_limit.pressed():
            self.move_arm(ArmDirection.DOWN, move_speed)

    def move_arm_to(self, position, move_speed):
        # Get the target position ratio
        target_arm_position_ratio = 0.0
        if position == ArmPosition.TOP_ENDSTOP:
            target_arm_position_ratio = 1.0
        elif position == ArmPosition.MIDDLE:
            target_arm_position_ratio = ARM_MIDDLE_POSITION_RATIO
        elif position == ArmPosition.DROP_OFF:
            target_arm_position_ratio = ARM_DROP_OFF_POSITION_RATIO

        # Calculate the target position in the arm step range
        target_arm_position = int(self.arm_total_steps * target_arm_position_ratio)
        
        # Step movement direction
        move_direction = ArmDirection.DOWN
        move_steps = self.current_arm_position - target_arm_position
        if move_steps < 0:
            move_direction = ArmDirection.UP
            move_steps = abs(move_steps)

        # Apply the movement
        for _ in range(move_steps):
            self.move_arm(move_direction, move_speed)

        # Apply recalibration of the arm
        self.recalibrate(move_speed)

        # Update the current arm position
        self.current_arm_position = target_arm_position

    def close_hand(self, move_speed):
        for _ in range(GRIP_STRENGTH):
            self.move_hand(HandAction.CLOSE, move_speed)

    def open_hand(self, move_speed):
        #while not self.end_stop_hand_open_limit.pressed():
        for _ in range(GRIP_STRENGTH):
            self.move_hand(HandAction.OPEN, move_speed)
