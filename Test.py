import time
from CurrentSensor import *
from Actions import *
from Motor import *
import threading
import signal
import RPi.GPIO as GPIO
import sys
import numpy as np
import plotext as plt

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

actions = CubertActions(motor)

light_on = False

current = []


_run_thread_1 = True

def check_light():

    global current
    global light_on
    global _run_thread_1
    
    while _run_thread_1:
        current.append(sensor.getChannelCurrent(CurrentChannel.BASE_LIGHT))


def spin_base():
    actions.rotateCube(BaseRotation.HALF, Direction.CCW)

lightThread = threading.Thread(target=check_light)
baseThread = threading.Thread(target=spin_base)

def sigint_handler(sig, frame):
    global _run_thread_1
    global actions
    global sensor
    global motor

    _run_thread_1 = False

    lightThread.join()
    baseThread.join()

    del actions
    del motor
    del sensor

    GPIO.cleanup()
    sys.exit(0)

if __name__ == '__main__':
    print("Running Test Sciprt")

    signal.signal(signal.SIGINT, sigint_handler)

    motor.spinBase(BaseRotation.QUARTER, Direction.CCW, 50)

    lightThread.start()
    
    motor.spinBase(BaseRotation.FULL, Direction.CCW, 1)

    time.sleep(0.001)

    _run_thread_1 = False

    lightThread.join()

    # plot stuff
    plt.scatter(np.array(current))
    plt.show()

