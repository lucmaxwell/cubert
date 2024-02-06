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

class MotorType(Enum):
    BASE = 0
    LEFT = 1
    RIGHT = 2

def get_step_delay(velocity):
    v = min(velocity, 200)
    x = MIN_SPEED + v * (MAX_SPEED - MIN_SPEED) / 100
    delay_duration = 1 / (0.0003 * x) / 10
    return round(delay_duration) / 1_000_000


class CubertMotor:

    _USE_UART = False


    _GEAR_RATIO     = 6



    def __init__(self, enable_pin, step_pin_list, dir_pin_list):

        self.tmc_base   = TMC_2209(enable_pin, step_pin_list[0], dir_pin_list[0],
                                   driver_address=0)

        self.tmc_left   = TMC_2209(pin_step=step_pin_list[1], pin_dir=dir_pin_list[1],
                                   driver_address=1)
        self.tmc_right  = TMC_2209(pin_step=step_pin_list[2], pin_dir=dir_pin_list[2],
                                   driver_address=2)

        self.tmc_list = [self.tmc_base, self.tmc_left, self.tmc_right]



    def __del__(self):
        self.disable()

        for tmc in self.tmc_list:
            del(tmc)

    def enable(self):
        self.tmc_base.set_motor_enabled(True)

    def disable(self):
        self.tmc_base.set_motor_enabled(False)

    def stop(self):
        for tmc in self.tmc_list:
            tmc.stop()

    # def home(self):

    def spinBase(self, degrees_to_rotate, move_direction, move_speed, degrees_to_correct=0, acceleration=0):
        revolutions = _GEAR_RATIO * degrees_to_rotate  / 360.0
        correction  = _GEAR_RATIO * degrees_to_correct / 360.0

        if move_direction == MotorSpin.COUNTER_CLOCKWISE:
            revolutions *= -1
            correction  *= -1

        self.tmc_base.set_vactual_rpm(move_speed, revolutions=(revolutions+correction), acceleration=acceleration)

        if abs(correction) > 0:
            self.tmc_base.set_vactual_rpm(move_speed, revolutions=-1*correction, acceleration=acceleration)


    def moveArm(self, distance_to_raise, move_speed, acceleration=0):
        return


    def step(self, steps, direction:MotorSpin, motor:MotorType, move_speed, correction_enable=False):
        # set step direction
        if direction == MotorSpin.COUNTER_CLOCKWISE:
            step_direction = -1
        else:
            step_direction = 1

        # Calculate the delay time of the pulse
        stepDelay = get_step_delay(move_speed)

        # Spin with given number of steps
        for _ in range(steps):
            self.tmc_list[motor].make_a_step()
            time.sleep(stepDelay)


if __name__ == '__main__':
    motor_en_pin = 26
    motor_step_pin = [27, 6, 19]
    motor_dir_pin = [17, 5, 13]

    # initialize motor
    motor = CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin)

    # Spin
    print("Running motor...")
    try:
        motor.enable()

        motor.step(1280, MotorSpin.CLOCKWISE, MotorType.BASE, 10)

        # print("Spinning CW 180")
        # motor.spinBase(180, MotorSpin.CLOCKWISE, 60)

        # print("Spinning CCW 180")
        # motor.spinBase(180, MotorSpin.COUNTER_CLOCKWISECLOCKWISE, 60)

        # print("Spinning CW 180 With Correction")
        # motor.spinBase(180, MotorSpin.CLOCKWISE, 60, 5)

        # print("Spinning CCW 180 With Correction")
        # motor.spinBase(180, MotorSpin.COUNTER_CLOCKWISE, 60, 5)

    except KeyboardInterrupt:
        pass
    finally:
        del motor

