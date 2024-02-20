import time
from enum import IntEnum
import INA3221.SDL_Pi_INA3221 as INA3221
import threading

class CurrentChannel(IntEnum):
    LEFT_MOTOR  = 1
    RIGHT_MOTOR = 2
    BASE_LIGHT  = 3

# define globale vairables
MOTOR_SKIPPED = False



class CubertCurrentSensor():

    run_gripper_monitor = False
    
    _current_threshold = 100

    def __init__(self):
        self.sensor = INA3221.SDL_Pi_INA3221(addr=0x40)

        self.run_gripper_monitor = True

        self._left_motor_monitor = threading.Thread(target=monitor_grip_current, args=(self, CurrentChannel.LEFT_MOTOR))
        self._right_motor_monitor = threading.Thread(target=monitor_grip_current, args=(self, CurrentChannel.RIGHT_MOTOR))

    def __del__(self):
        self.run_gripper_monitor = False

        self._left_motor_monitor.join()
        self._right_motor_monitor.join()

    def getChannelCurrent(self, channel:CurrentChannel):
        return self.sensor.getCurrent_mA(channel)
    
    
def monitor_grip_current(sensor:CubertCurrentSensor, channel:CurrentChannel):
    
    curr_reading = 0
    prev_reading = 0

    global MOTOR_SKIPPED

    while sensor.run_gripper_monitor:
        prev_reading = curr_reading
        curr_reading = sensor.getChannelCurrent(channel)

        print(curr_reading)

        if abs(prev_reading - curr_reading) > 100:#sensor._current_threshold:
            MOTOR_SKIPPED = True

if __name__ == '__main__':
    print("Running Current Sensor Test")

    sensor = CubertCurrentSensor()

    while True:
        print("Base Light Current is %fmA" % sensor.getChannelCurrent(CurrentChannel.BASE_LIGHT))