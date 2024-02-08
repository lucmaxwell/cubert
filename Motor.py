import RPi.GPIO as GPIO
import signal
import sys
import time
from enum import Enum, IntEnum
from TMC2209MotorLib.src.TMC_2209.TMC_2209_StepperDriver import *


# Parameter
MAX_SPEED = 3.3 # DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
MIN_SPEED = 0.000001

class Direction(IntEnum):
    CCW = 0
    CW = 1

class GripperDirection(Enum):
    UP = 0
    DOWN = 1
    OPEN = 2
    CLOSE = 3

class MotorType(IntEnum):
    BASE = 0
    LEFT = 1
    RIGHT = 2

def get_step_delay(velocity):
    v = min(velocity, 200)
    x = MIN_SPEED + v * (MAX_SPEED - MIN_SPEED) / 100
    delay_duration = 1 / (0.0003 * x) / 10
    return round(delay_duration) / 1_000_000

def sigint_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)


class CubertMotor:

    _USE_UART = False


    _GEAR_RATIO     = 6

    _top_endstop_pressed        = False
    _bottom_endstop_pressed     = False
    _gripper_endstop_pressed    = False

    def __init__(self, enable_pin, step_pin_list, dir_pin_list, top_end_pin, bottom_end_pin, grip_end_pin):

        # setup GPIO
        GPIO.setmode(GPIO.BCM)

        # setup base motor
        self.tmc_base   = TMC_2209(enable_pin, step_pin_list[0], dir_pin_list[0],
                                   driver_address=0)

        # setup gripper motors
        self.tmc_left   = TMC_2209(pin_step=step_pin_list[1], pin_dir=dir_pin_list[1],
                                   driver_address=1)
        self.tmc_right  = TMC_2209(pin_step=step_pin_list[2], pin_dir=dir_pin_list[2],
                                   driver_address=2)

        # used to index motors
        self.tmc_list = [self.tmc_base, self.tmc_left, self.tmc_right]

        # store enstop pins
        self._top_end_pin       = top_end_pin
        self._bottom_end_pin    = bottom_end_pin
        self._grip_end_pin      = grip_end_pin

        # setup enstops
        GPIO.setup(   top_end_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(bottom_end_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(  grip_end_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # setup endstop interrupts
        GPIO.add_event_detect(top_end_pin, GPIO.BOTH,
                              callback=self.top_endstop_callback)
        GPIO.add_event_detect(bottom_end_pin, GPIO.BOTH,
                              callback=self.bottom_endstop_callback)
        GPIO.add_event_detect(grip_end_pin, GPIO.BOTH,
                              callback=self.gripper_endstop_callback)




    def __del__(self):
        self.disable()

        for tmc in self.tmc_list:
            del(tmc)

        GPIO.cleanup()



    def enable(self):
        self.tmc_base.set_motor_enabled(True)

    def disable(self):
        self.tmc_base.set_motor_enabled(False)

    def stop(self):
        for tmc in self.tmc_list:
            tmc.stop()


    def top_endstop_callback(self, channel):
        if not GPIO.input(self._top_end_pin) and not self._top_endstop_pressed:
            self._top_endstop_pressed = True
            print("Top Endstop Pressed")

        elif GPIO.input(self._top_end_pin) and self._top_endstop_pressed:
            self._top_endstop_pressed = False
            print("Top Endstop Released")

    def bottom_endstop_callback(self, channel):
        if not GPIO.input(self._bottom_end_pin) and not self._bottom_endstop_pressed:
            self._bottom_endstop_pressed = True
            print("Bottom Endstop Pressed")

        elif GPIO.input(self._bottom_end_pin) and self._bottom_endstop_pressed:
            self._bottom_endstop_pressed = False
            print("Bottom Endstop Released")

    def gripper_endstop_callback(self, channel):
        if not GPIO.input(self._grip_end_pin) and not self._gripper_endstop_pressed:
            self._gripper_endstop_pressed = True
            print("Gripper Endstop Pressed")

        elif GPIO.input(self._grip_end_pin) and self._gripper_endstop_pressed:
            self._gripper_endstop_pressed = False
            print("Gripper Endstop Released")


    def spinBase(self, degrees_to_rotate, move_direction, move_speed, degrees_to_correct=0, acceleration=0):
        revolutions = _GEAR_RATIO * degrees_to_rotate  / 360.0
        correction  = _GEAR_RATIO * degrees_to_correct / 360.0

        if move_direction == Direction.CCW:
            revolutions *= -1
            correction  *= -1

        self.tmc_base.set_vactual_rpm(move_speed, revolutions=(revolutions+correction), acceleration=acceleration)

        if abs(correction) > 0:
            self.tmc_base.set_vactual_rpm(move_speed, revolutions=-1*correction, acceleration=acceleration)


    def stepGripper(self, steps, direction:GripperDirection, move_speed):

        # tracks steps completed
        steps_done = 0

        # calculate step delay
        step_delay = get_step_delay(move_speed)

        # set step directions
        if direction == GripperDirection.UP:
            self.tmc_left.set_direction_pin(Direction.CCW)
            self.tmc_right.set_direction_pin(Direction.CW)

            while (not self._top_endstop_pressed) and steps_done < steps:
                self.tmc_left.make_a_step()
                self.tmc_right.make_a_step()
                steps_done += 1
                time.sleep(step_delay)

        elif direction == GripperDirection.DOWN:
            self.tmc_left.set_direction_pin(Direction.CW)
            self.tmc_right.set_direction_pin(Direction.CCW)

            while (not self._bottom_endstop_pressed) and steps_done < steps:
                self.tmc_left.make_a_step()
                self.tmc_right.make_a_step()
                steps_done += 1
                time.sleep(step_delay)
        
        elif direction == GripperDirection.OPEN:
            self.tmc_left.set_direction_pin(Direction.CW)
            self.tmc_right.set_direction_pin(Direction.CW)

            while (not self._gripper_endstop_pressed) and steps_done < steps:
                self.tmc_left.make_a_step()
                self.tmc_right.make_a_step()
                steps_done += 1
                time.sleep(step_delay)
        
        elif direction == GripperDirection.CLOSE:
            self.tmc_left.set_direction_pin(Direction.CCW)
            self.tmc_right.set_direction_pin(Direction.CCW)

            while steps_done < steps:
                self.tmc_left.make_a_step()
                self.tmc_right.make_a_step()
                steps_done += 1
                time.sleep(step_delay)

        else:
            print("ERROR: Direction does not exist!")

        # # calculate step delay
        # step_delay = get_step_delay(move_speed)

        # # spin for given number of steps
        # for _ in range(steps):
        #     self.tmc_left.make_a_step()
        #     self.tmc_right.make_a_step()
        #     time.sleep(step_delay)

    def stepBase(self, steps, direction:Direction, move_speed):
        # set step direction
        self.tmc_base.set_direction_pin(direction)

        # calculate step delay
        step_delay = get_step_delay(move_speed)

        # spind for given number of steps
        for _ in range(steps):
            self.tmc_base.make_a_step()
            time.sleep(step_delay)

    def step(self, steps, direction:Direction, motor:MotorType, move_speed):
        # set step direction
        self.tmc_list[motor].set_direction_pin(direction)

        # Calculate the delay time of the pulse
        stepDelay = get_step_delay(move_speed)

        # Spin with given number of steps
        for _ in range(steps):
            self.tmc_list[motor].make_a_step()
            time.sleep(stepDelay)


if __name__ == '__main__':
    motor_en_pin = 26
    motor_step_pin = [27, 6, 19]
    motor_dir_pin = [17, 5, 13]

    # End stop for arm
    end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
    end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
    end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

    # initialize motor
    motor = CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin)

    signal.signal(signal.SIGINT, sigint_handler)

    # Spin
    print("Running motor...")
    try:
        motor.enable()

        motor.step(19200, Direction.CW, MotorType.BASE, 10)

        motor.stepBase(19200, Direction.CCW, 60)

        motor.stepGripper(1000, GripperDirection.UP, 10)
        time.sleep(1)
        motor.stepGripper(1000, GripperDirection.DOWN, 10)
        time.sleep(1)
        motor.stepGripper(1000, GripperDirection.CLOSE, 10)
        time.sleep(1)
        motor.stepGripper(1000, GripperDirection.OPEN, 10)

        print("Testing Complete!")

        while True:
            # do nothing
            time.sleep(1)

        # print("Spinning CW 180")
        # motor.spinBase(180, MotorSpin.CLOCKWISE, 60)

        # print("Spinning CCW 180")
        # motor.spinBase(180, MotorSpin.COUNTER_CLOCKWISECLOCKWISE, 60)

        # print("Spinning CW 180 With Correction")
        # motor.spinBase(180, MotorSpin.CLOCKWISE, 60, 5)

        # print("Spinning CCW 180 With Correction")
        # motor.spinBase(180, MotorSpin.COUNTER_CLOCKWISE, 60, 5)

    except KeyboardInterrupt:
        pass
    finally:
        del motor

