import time
from enum import IntEnum
import INA3221.SDL_Pi_INA3221 as INA3221

class CurrentChannel(IntEnum):
    LEFT_MOTOR  = 1
    RIGHT_MOTOR = 2
    BASE_LIGHT  = 3

class CubertCurrentSensor():

    def __init__(self):
        self.sensor = INA3221.SDL_Pi_INA3221(addr=0x40)

    def getChannelCurrent(self, channel:CurrentChannel):
        return self.sensor.getCurrent_mA(channel)
    

if __name__ == '__main__':
    print("Running Current Sensor Test")

    sensor = CubertCurrentSensor()

    while True:
        print("Base Light Current is %fmA" % sensor.getChannelCurrent(CurrentChannel.BASE_LIGHT))