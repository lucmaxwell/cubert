import RPi.GPIO as GPIO
import time
from enum import Enum


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
    def __init__(self, enable_pin, pin_list):
        self.step_pin = pin_list[0]
        self.dir_pin = pin_list[1]
        self.enable_pin = enable_pin

    def enable(self):
        GPIO.output(self.enable_pin, GPIO.LOW)

    def disable(self):
        GPIO.output(self.enable_pin, GPIO.HIGH)

    def step(self, steps, direction, move_speed, correction_enable=False):
        # Write the spin direction
        GPIO.output(self.dir_pin, GPIO.HIGH if direction == MotorSpin.CLOCKWISE else GPIO.LOW)

        # Calculate the delay time of the pulse
        stepDelay = get_step_delay(move_speed)

        # Spin with given number of steps
        for _ in range(steps):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(stepDelay)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(stepDelay)
