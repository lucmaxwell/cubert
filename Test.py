import time
from CurrentSensor import *
from Actions import *
from Motor import *
import threading

# Motor Pins
motor_en_pin = 26
motor_step_pin = [27, 6, 19]
motor_dir_pin = [17, 5, 13]

# End stop for arm
end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

sensor = CubertCurrentSensor()

motor = CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin, sensor = CurrentSensor.CurrentSensor())

actions = CubertActions(motor)

def thread_function(sensor:CubertCurrentSensor):
    print("Starting Current Sensing")
    
    while True:
        if sensor.getChannelCurrent(CurrentChannel.BASE_LIGHT) > 100:
            print("Base Light On!")

if __name__ == '__main__':
    print("Running Test Sciprt")

    thread = threading.Thread(target=thread_function)

    actions.rotateCube(BaseRotation.HALF, Direction.CCW)

    del actions
    del motor

