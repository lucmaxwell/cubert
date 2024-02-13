import time
from CurrentSensor import *
from Motor import *
import threading
import signal
import RPi.GPIO as GPIO
import sys
import numpy as np
import plotext as plt
import Vision
from Actions import *

# Motor Pins
motor_en_pin = 26
motor_step_pin = [27, 6, 19]
motor_dir_pin = [17, 5, 13]

# End stop for arm
end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

sensor = CubertCurrentSensor()

motor = CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin, sensor)

vision = Vision.CubertVision()

actions = CubertActions(motor, vision)

light_on = False

current_base = []
current_left = []
current_right = []


_run_thread_1 = True

def check_light():

    global current_base
    global current_left
    global current_right
    global light_on
    global _run_thread_1
    
    while _run_thread_1:
        # current_base.append(sensor.getChannelCurrent(CurrentChannel.BASE_LIGHT))
        current_left.append(sensor.getChannelCurrent(CurrentChannel.LEFT_MOTOR))
        current_right.append(sensor.getChannelCurrent(CurrentChannel.RIGHT_MOTOR))


def spin_base():
    actions.rotateCube(BaseRotation.HALF, Direction.CCW)

currentThread = threading.Thread(target=check_light)
baseThread = threading.Thread(target=spin_base)

def sigint_handler(sig, frame):
    global _run_thread_1
    global actions
    global sensor
    global motor

    _run_thread_1 = False

    currentThread.join()
    baseThread.join()

    del actions
    del motor
    del sensor

    GPIO.cleanup()
    sys.exit(0)

if __name__ == '__main__':
    print("Running Test Sciprt")

    actions.solve(True)

    del actions
    del motor
    del sensor

    GPIO.cleanup()
    sys.exit(0)

