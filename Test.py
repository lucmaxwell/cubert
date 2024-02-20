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
import Solver

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

solver = Solver.Solver()

actions = CubertActions(motor, vision, solver)

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
    global actions
    global sensor
    global motor

    del actions
    del motor
    del sensor

    GPIO.cleanup()
    sys.exit(0)

def worker(selection):
    if selection == '0': # Single solve
        time.sleep(5)
        actions.solve(True)
        time.sleep(15)

    elif selection == '1': # Single scramble
        actions.scramble(13)

    elif selection == '2': # Endless scramble + solve
        while True:
            actions.scramble(13)
            time.sleep(5)
            motor.homeLight()
            time.sleep(5)
            actions.solve(True)
            time.sleep(15)

    elif selection == 3: # Take an image
        cube, mask = actions.getAllImages(True)
        vision.writeImage("testingImage.png", cube)
        vision.writeImage("testingmask.png", mask)

_PANIC_BUTTON_PIN = 4

if __name__ == '__main__':
    print("Running Test Script")

    # Setup panic pin
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(_PANIC_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    signal.signal(signal.SIGINT, sigint_handler)
    print()
    print("==========================================")
    print("==========================================")
    print()
    
    print("0: Single solve")
    print("1: Scramble")
    print("2: Endless Scramble + solve")
    print("3: Take picture, save to ./images")
    selection = input("Select an option: ")

    # Set up the work thread
    worker_thread = threading.Thread(target=worker, args=(selection))
    worker_thread.daemon = True
    worker_thread.start()

    # Run the program until panic
    panic = False
    _PANIC_BUTTON_PIN = 4
    while not panic:
        # Take a reading of the panic button
        panic = (GPIO.input(_PANIC_BUTTON_PIN) == GPIO.LOW)

        # End the worker thread
        if panic:
            sys.exit("Program terminated due to panic button pressed.")

        time.sleep(0.2)

    del actions
    del motor
    del sensor

    GPIO.cleanup()
    sys.exit(0)

