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
        self.enable_pin = enable_pin
        self.step_pin = pin_list[0]
        self.dir_pin = pin_list[1]

    def enable(self):
        print("Enable pin")
        GPIO.output(self.enable_pin, GPIO.LOW)

    def disable(self):
        GPIO.output(self.enable_pin, GPIO.HIGH)

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
    motor_en_pin = 5
    motor_step_pin = 27
    motor_dir_pin = 22

    # GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(motor_en_pin, GPIO.OUT)
    GPIO.setup(motor_step_pin, GPIO.OUT)
    GPIO.setup(motor_dir_pin, GPIO.OUT)

    # Create the motor wrapper
    pin_list = [motor_step_pin, motor_dir_pin]
    motor = CubertMotor(motor_en_pin, pin_list)

    # Spin
    print("Running motor...")
    try:
        GPIO.output(motor_en_pin, GPIO.LOW)
        while True:
            motor.step(1, MotorSpin.COUNTER_CLOCKWISE, 60)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
