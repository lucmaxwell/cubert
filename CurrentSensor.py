import time
from enum import IntEnum
import INA3221.SDL_Pi_INA3221 as INA3221
import threading
import numpy as np
import ctypes
# from Motor import MotorType

libc = ctypes.CDLL('libc.so.6')

class MotorType(IntEnum):
    """Identifies the motor to access"""
    BASE    = 0
    LEFT    = 1
    RIGHT   = 2

class CurrentChannel(IntEnum):
    LEFT_MOTOR  = 1
    RIGHT_MOTOR = 2
    BASE_LIGHT  = 3

# define globale vairables
MOTOR_SKIPPED = threading.Event()
MOTOR_SKIPPED_LOCK = threading.Lock()

class CubertCurrentSensor():

    run_gripper_monitor = threading.Event()
    
    _current_threshold = 2000

    _left_log_list = [[], []]
    _right_log_list = [[], []]

    _left_monitor_list = [[], []]
    _right_monitor_list = [[], []]

    _left_log_lock = threading.Lock()
    _right_log_lock = threading.Lock()

    def __init__(self):
        MOTOR_SKIPPED.clear() # set to false

        self.sensor = INA3221.SDL_Pi_INA3221(addr=0x40)

        self.run_gripper_monitor.set()

        self._left_motor_monitor = threading.Thread(target=monitor_grip_current, args=(self, CurrentChannel.LEFT_MOTOR, self._left_log_list, self._left_log_lock))
        self._right_motor_monitor = threading.Thread(target=monitor_grip_current, args=(self, CurrentChannel.RIGHT_MOTOR, self._right_log_list, self._right_log_lock))

        self._left_motor_monitor.start()
        self._right_motor_monitor.start()

    def __del__(self):
        print("Deleting Sensor")
        self.run_gripper_monitor.clear()

        print("Terminating Threads")
        self._left_motor_monitor.join()
        self._right_motor_monitor.join()

        del self.sensor

        print("Saving Data")
        np.save("./logging/left_motor_current_reading.npy", np.array(self._left_log_list[0]))
        np.save("./logging/right_motor_current_reading.npy", np.array(self._right_log_list[0]))
        np.save("./logging/left_motor_current_delta.npy", np.array(self._left_log_list[1]))
        np.save("./logging/right_motor_current_delta.npy", np.array(self._right_log_list[1]))

        np.save("./logging/left_motor_current_reading_step.npy", np.array(self._left_monitor_list[0]))
        np.save("./logging/right_motor_current_reading_step.npy", np.array(self._right_monitor_list[0]))
        np.save("./logging/left_motor_current_delta_step.npy", np.array(self._left_monitor_list[1]))
        np.save("./logging/right_motor_current_delta_step.npy", np.array(self._right_monitor_list[1]))

        print("Sensor Deleted")

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

    def logCurrent(self, motor:MotorType):
        if motor == MotorType.LEFT:

            self._left_monitor_list[0].append(self.sensor.getCurrent_mA(CurrentChannel.LEFT_MOTOR))

            if len(self._left_monitor_list[0]) > 1:
                i = len(self._left_monitor_list[0]) - 1

                self._left_monitor_list[1].append(self._left_monitor_list[0][i] - self._left_monitor_list[0][i-1])

        elif motor == MotorType.RIGHT:
            self._right_monitor_list[0].append(self.sensor.getCurrent_mA(CurrentChannel.LEFT_MOTOR))

            if len(self._right_monitor_list[0]) > 1:
                i = len(self._right_monitor_list[0]) - 1

                self._right_monitor_list[1].append(self._right_monitor_list[0][i] - self._right_monitor_list[0][i-1])
    
    
def monitor_grip_current(sensor:CubertCurrentSensor, channel:CurrentChannel, log_list, log_lock:threading.Lock):
    
    conversion_time = 160 # time to wait for new sample

    curr_reading = 0
    prev_reading = 0

    while sensor.run_gripper_monitor.isSet():
        prev_reading = curr_reading
        curr_reading = sensor.getChannelCurrent(channel)

        delta = prev_reading - curr_reading

        log_lock.acquire()
        log_list[0].append(curr_reading)
        log_list[1].append(delta)
        log_lock.release()

        # print(curr_reading)

        if abs(delta) > sensor._current_threshold:
            print("Motor Skipped!")
            MOTOR_SKIPPED_LOCK.acquire()
            MOTOR_SKIPPED.set()
            MOTOR_SKIPPED_LOCK.release()

        # wait for next conversion
        libc.usleep(conversion_time)

if __name__ == '__main__':
    print("Running Current Sensor Test")

    sensor = CubertCurrentSensor()

    while True:
        print("Base Light Current is %fmA" % sensor.getChannelCurrent(CurrentChannel.BASE_LIGHT))