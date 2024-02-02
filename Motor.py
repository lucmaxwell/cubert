import RPi.GPIO as GPIO
import time
from enum import Enum
from TMC2209MotorLib.src.TMC_2209.TMC_2209_StepperDriver import *


# Parameter
MAX_SPEED = 3.3 # DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
MIN_SPEED = 0.000001


class MotorSpin(Enum):
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1


def get_step_delay(velocity):
    v = min(velocity, 200)
    x = MIN_SPEED + v * (MAX_SPEED - MIN_SPEED) / 100
    delay_duration = 1 / (0.0003 * x) / 10
    return round(delay_duration) / 1_000_000


class CubertMotor:
    def __init__(self, enable_pin, step_pin_list, dir_pin_list):

        self.tmc_base   = TMC_2209(enable_pin, step_pin_list[0], dir_pin_list[0],
                                   driver_address=0)

        self.tmc_left   = TMC_2209(pin_step=step_pin_list[1], pin_dir=dir_pin_list[1],
                                   driver_address=1)
        self.tmc_right  = TMC_2209(pin_step=step_pin_list[2], pin_dir=dir_pin_list[2],
                                   driver_address=2)

        self.tmc_list = [self.tmc_base, self.tmc_left, self.tmc_right]

        # self.base_steps_per_rev     = self.tmc_base.get_steps_per_rev()
        # self.left_steps_per_rev     = self.tmc_left.get_steps_per_rev()
        # self.right_steps_per_rev    = self.tmc_right.get_steps_per_rev()

        # self.enable_pin = enable_pin
        # self.step_pin = pin_list[0]
        # self.dir_pin = pin_list[1]

    def __del__(self):
        self.disable()

        for tmc in self.tmc_list:
            del(tmc)

    def enable(self):
        self.tmc_base.set_motor_enabled(True)
        # GPIO.output(self.enable_pin, GPIO.LOW)

    def disable(self):
        self.tmc_base.set_motor_enabled(False)
        # GPIO.output(self.enable_pin, GPIO.HIGH)

    def spinBase(self, degrees_to_rotate, move_direction, move_speed, degrees_to_correct=0):
        revolutions = degrees_to_rotate  / 360.0
        correction  = degrees_to_correct / 360.0

        if move_direction == MotorSpin.COUNTER_CLOCKWISE:
            revolutions *= -1
            correction  *= -1

        self.tmc_base.set_vactual_rpm(move_speed, revolutions=(revolutions+correction))

        if abs(correction) > 0:
            self.tmc_base.set_vactual_rpm(move_speed, revolutions=correction)

        


    def step(self, steps, direction, move_speed, correction_enable=False):
        # Write the spin direction
        GPIO.output(self.dir_pin, GPIO.LOW if direction == MotorSpin.CLOCKWISE else GPIO.HIGH)

        # Calculate the delay time of the pulse
        stepDelay = get_step_delay(move_speed)

        # Spin with given number of steps
        for _ in range(steps):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(stepDelay)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(stepDelay)


if __name__ == '__main__':
    motor_en_pin = 26
    motor_step_pin = [27, 6, 19]
    motor_dir_pin = [17, 5, 13]

    # # GPIO
    GPIO.setmode(GPIO.BCM)
    # GPIO.setup(motor_en_pin, GPIO.OUT)
    # GPIO.setup(motor_step_pin, GPIO.OUT)
    # GPIO.setup(motor_dir_pin, GPIO.OUT)

    # Create the motor wrapper
    # pin_list = [motor_step_pin, motor_dir_pin]
    motor = CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin)

    # Spin
    print("Running motor...")
    try:
        motor.enable()

        print("Spinning CW 180")
        motor.spinBase(180, MotorSpin.CLOCKWISE, 60)

        print("Spinning CCW 180")
        motor.spinBase(180, MotorSpin.COUNTER_CLOCKWISECLOCKWISE, 60)

        print("Spinning CW 180 With Correction")
        motor.spinBase(180, MotorSpin.CLOCKWISE, 60, 5)

        print("Spinning CCW 180 With Correction")
        motor.spinBase(180, MotorSpin.COUNTER_CLOCKWISE, 60, 5)

        # while True:
        #     motor.step(1, MotorSpin.COUNTER_CLOCKWISE, 120)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        del motor

