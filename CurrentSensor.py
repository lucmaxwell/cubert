import time
from enum import IntEnum
import INA3221.SDL_Pi_INA3221 as INA3221
import threading
import numpy as np

class CurrentChannel(IntEnum):
    LEFT_MOTOR  = 1
    RIGHT_MOTOR = 2
    BASE_LIGHT  = 3

# define globale vairables
MOTOR_SKIPPED = threading.Event()
MOTOR_SKIPPED_LOCK = threading.Lock()

class CubertCurrentSensor():

    run_gripper_monitor = threading.Event()
    
    _current_threshold = 15

    _left_log_list = []
    _right_log_list = []

    def __init__(self):
        MOTOR_SKIPPED.clear() # set to false

        self.sensor = INA3221.SDL_Pi_INA3221(addr=0x40)

        self.run_gripper_monitor.set()

        self._left_motor_monitor = threading.Thread(target=monitor_grip_current, args=(self, CurrentChannel.LEFT_MOTOR, self._left_log_list))
        self._right_motor_monitor = threading.Thread(target=monitor_grip_current, args=(self, CurrentChannel.RIGHT_MOTOR, self._right_log_list))

        self._left_motor_monitor.start()
        self._right_motor_monitor.start()

    def __del__(self):
        self.run_gripper_monitor.clear()

        self._left_motor_monitor.join()
        self._right_motor_monitor.join()

        np.save("./logging/left_motor_current.npy", np.array(self._left_log_list))
        np.save("./logging/right_motor_current.npy", np.array(self._right_log_list))

    def getChannelCurrent(self, channel:CurrentChannel):
        return self.sensor.getCurrent_mA(channel)
    
    def getMotorSkipped(self):
        MOTOR_SKIPPED_LOCK.acquire()
        val = MOTOR_SKIPPED.isSet()
        MOTOR_SKIPPED_LOCK.release()
        return val
    
    def clearSkipFlag(self):
        MOTOR_SKIPPED_LOCK.acquire()
        MOTOR_SKIPPED.clear()
        MOTOR_SKIPPED_LOCK.release()
    
    
def monitor_grip_current(sensor:CubertCurrentSensor, channel:CurrentChannel, log_list):
    
    curr_reading = 0
    prev_reading = 0

    while sensor.run_gripper_monitor.isSet():
        prev_reading = curr_reading
        curr_reading = sensor.getChannelCurrent(channel)

        delta = prev_reading - curr_reading

        log_list.append(delta)

        # print(curr_reading)

        if abs(delta) > sensor._current_threshold:
            print("Motor Skipped!")
            MOTOR_SKIPPED_LOCK.acquire()
            MOTOR_SKIPPED.set()
            MOTOR_SKIPPED_LOCK.release()

if __name__ == '__main__':
    print("Running Current Sensor Test")

    sensor = CubertCurrentSensor()

    while True:
        print("Base Light Current is %fmA" % sensor.getChannelCurrent(CurrentChannel.BASE_LIGHT))