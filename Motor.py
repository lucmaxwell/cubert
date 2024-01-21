import RPi.GPIO as GPIO
import time
from enum import Enum


class MotorSpin(Enum):
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1


class CubertMotor:
    def __init__(self, enable_pin, pin_list):
        self.step_pin = pin_list[0]
        self.dir_pin = pin_list[1]
        self.enable_pin = enable_pin

    def enable(self):
        GPIO.output(self.enable_pin, GPIO.LOW)

    def disable(self):
        GPIO.output(self.enable_pin, GPIO.HIGH)

    def step(self, steps, direction, delay=0.01, correction_enable=False):
        GPIO.output(self.dir_pin, GPIO.HIGH if direction == MotorSpin.CLOCKWISE else GPIO.LOW)
        for _ in range(steps):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(delay)
