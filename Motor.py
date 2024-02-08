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
    CW  = 1

class GripperDirection(Enum):
    UP      = 0
    DOWN    = 1
    OPEN    = 2
    CLOSE   = 3

class MotorType(IntEnum):
    BASE    = 0
    LEFT    = 1
    RIGHT   = 2

class GripperPosition(Enum):
    UNKNOWN = 0
    TOP     = 1
    BOTTOM  = 2
    MIDDLE  = 3

def get_step_delay(velocity):
    v = min(velocity, 200)
    x = MIN_SPEED + v * (MAX_SPEED - MIN_SPEED) / 100
    delay_duration = 1 / (0.0003 * x) / 10
    return round(delay_duration) / 1_000_000

def sigint_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)


class CubertMotor:

    # class constants
    _USE_UART = False

    _ACTUAL_STEPS   = 400   # number of steps in motor
    _MICROSTEPS     = 8     # set microstep resolution
    _GEAR_RATIO     = 6     # cube base gear ratio

    _MAX_CURRENT    = 700   # max current draw of motors in mA

    _DISTANCE_FROM_BOTTOM_TO_TOP    = 64    # distance from gripper travel bottom to top in mm

    _DEFAULT_MOVE_SPEED     = 10

    # derived class constants
    _STEPS_PER_REV  = _ACTUAL_STEPS * _MICROSTEPS # number of steps per revolution

    # determines if enstops are pressed
    _top_endstop_pressed        = False
    _bottom_endstop_pressed     = False
    _gripper_endstop_pressed    = False

    # class variables
    _steps_per_mm       = -1
    _steps_to_top       = -1
    _steps_to_bottom    = -1
    _steps_total_travel = -1

    _current_gripper_pos    = GripperPosition.UNKNOWN
    _current_hand_pos       = -1


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

        # setup motors
        for tmc in self.tmc_list:
            tmc.set_current(700)
            tmc.set_microstepping_resolution(self._MICROSTEPS)
            tmc.set_interpolation(True)

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


    def home(self):
        self.homeGripper()
        self.homeBase()

    def homeGripper(self):
        self.gripperMoveToPos(GripperPosition.BOTTOM, 20)
        self._steps_total_travel = self.gripperMoveToPos(GripperPosition.TOP, 20)
        self.gripperMoveToPos(GripperPosition.MIDDLE)

    def homeBase(self):
        return



    def top_endstop_callback(self, channel):
        if not GPIO.input(self._top_end_pin) and not self._top_endstop_pressed:
            self._top_endstop_pressed = True
            self._current_gripper_pos = GripperPosition.TOP
            print("Top Endstop Pressed")

        elif GPIO.input(self._top_end_pin) and self._top_endstop_pressed:
            self._top_endstop_pressed = False
            print("Top Endstop Released")

    def bottom_endstop_callback(self, channel):
        if not GPIO.input(self._bottom_end_pin) and not self._bottom_endstop_pressed:
            self._bottom_endstop_pressed = True
            self._current_gripper_pos = GripperPosition.BOTTOM
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


    # depreciated spin base function
    def spinBaseDep(self, degrees_to_rotate, move_direction, move_speed, degrees_to_correct=0, acceleration=0):
        revolutions = _GEAR_RATIO * degrees_to_rotate  / 360.0
        correction  = _GEAR_RATIO * degrees_to_correct / 360.0

        if move_direction == Direction.CCW:
            revolutions *= -1
            correction  *= -1

        self.tmc_base.set_vactual_rpm(move_speed, revolutions=(revolutions+correction), acceleration=acceleration)

        if abs(correction) > 0:
            self.tmc_base.set_vactual_rpm(move_speed, revolutions=-1*correction, acceleration=acceleration)


    def gripperMoveToPos(self, position:GripperPosition, move_speed=_DEFAULT_MOVE_SPEED):
        endstop_to_check = False
        steps            = sys.maxsize

        # tracks steps completed
        steps_done = 0

        # calculate step delay
        step_delay = get_step_delay(move_speed)

        print("Moving Gripper to {position}")

        # check direction to step
        if position == GripperPosition.TOP:

            endstop_to_check = self._top_endstop_pressed

            direction = GripperDirection.UP

        elif position == GripperPosition.BOTTOM:

            endstop_to_check = self._bottom_endstop_pressed

            direction = GripperDirection.DOWN
        
        elif position == GripperPosition.MIDDLE:

            endstop_to_check = self._gripper_endstop_pressed

            if self._current_gripper_pos == GripperPosition.TOP:
                direction = GripperDirection.DOWN
                steps = self._steps_total_travel/2

            if self._current_gripper_pos == GripperPosition.BOTTOM:
                direction = GripperDirection.UP
                steps = self._steps_total_travel/2

            if self._current_gripper_pos == GripperPosition.UNKNOWN:
                endstop_to_check = True # exit loop
                print("Position Currently Unknown: Cannot Determine Direction to Middle!")            

        while (not endstop_to_check) and steps_done < steps:
            self.stepGripper(direction)
            steps_done += 1
            time.sleep(step_delay)

        self._current_gripper_pos = position

        return steps_done
    
    def gripperMoveMM(self, mm_to_move, move_speed=_DEFAULT_MOVE_SPEED):
        return


    def moveGripper(self, steps, direction:GripperDirection, move_speed=_DEFAULT_MOVE_SPEED):

        endstop_to_check = False

        # tracks steps completed
        steps_done = 0

        # calculate step delay
        step_delay = get_step_delay(move_speed)

        # check direction to step
        if direction == GripperDirection.UP:
            print("Moving Gripper Up")

            endstop_to_check = self._top_endstop_pressed

        elif direction == GripperDirection.DOWN:
            print("Moving Gripper Down")

            endstop_to_check = self._bottom_endstop_pressed
        
        elif direction == GripperDirection.OPEN:
            print("Opening Gripper")

            endstop_to_check = self._gripper_endstop_pressed
        
        elif direction == GripperDirection.CLOSE:
            print("Closing Gripper")

        while (not endstop_to_check) and steps_done < steps:
            self.stepGripper(direction)
            steps_done += 1
            time.sleep(step_delay)

        return steps_done

        

    def moveBase(self, steps, direction:Direction, move_speed=_DEFAULT_MOVE_SPEED):
        # calculate step delay
        step_delay = get_step_delay(move_speed)

        # spin for given number of steps
        for _ in range(steps):
            self.stepBase(direction)
            time.sleep(step_delay)

    def move(self, steps, direction:Direction, motor:MotorType, move_speed=_DEFAULT_MOVE_SPEED):
        # Calculate the delay time of the pulse
        stepDelay = get_step_delay(move_speed)

        # Spin with given number of steps
        for _ in range(steps):
            self.step(direction, motor)
            time.sleep(stepDelay)


    def stepGripper(self, direction:GripperDirection):

        endstop_to_check = False

        # check direction to step
        if direction == GripperDirection.UP:
            self.tmc_left.set_direction_pin(Direction.CCW)
            self.tmc_right.set_direction_pin(Direction.CW)

            endstop_to_check = self._top_endstop_pressed

        elif direction == GripperDirection.DOWN:
            self.tmc_left.set_direction_pin(Direction.CW)
            self.tmc_right.set_direction_pin(Direction.CCW)

            endstop_to_check = self._bottom_endstop_pressed
        
        elif direction == GripperDirection.OPEN:
            self.tmc_left.set_direction_pin(Direction.CCW)
            self.tmc_right.set_direction_pin(Direction.CCW)

            endstop_to_check = self._gripper_endstop_pressed
        
        elif direction == GripperDirection.CLOSE:
            self.tmc_left.set_direction_pin(Direction.CW)
            self.tmc_right.set_direction_pin(Direction.CW)
            
        else:
            print("ERROR: Direction does not exist!")
            return -1

        # step gripper
        if not endstop_to_check:
            self.tmc_left.make_a_step()
            self.tmc_right.make_a_step()

    def stepBase(self, direction:Direction):
        # set step direction
        self.tmc_base.set_direction_pin(direction)

        # step base
        self.tmc_base.make_a_step()

    def step(self, direction:Direction, motor:MotorType):
        # set step direction
        self.tmc_list[motor].set_direction_pin(direction)

        # step motor
        self.tmc_list[motor].make_a_step()

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

        motor.move(19200, Direction.CW, MotorType.BASE, 10)

        motor.moveBase(19200, Direction.CCW, 60)

        motor.home()

        motor.moveGripper(10000, GripperDirection.DOWN, 10)
        time.sleep(1)
        motor.moveGripper(10000, GripperDirection.UP, 10)
        time.sleep(1)
        motor.moveGripper(200, GripperDirection.CLOSE, 10)
        time.sleep(1)
        motor.moveGripper(10000, GripperDirection.OPEN, 10)

        print("Testing Complete!")

        while True:
            # do nothing
            motor.moveGripper(10000, GripperDirection.DOWN, 40)
            time.sleep(1)
            motor.moveGripper(10000, GripperDirection.UP, 50)
            time.sleep(1)
            motor.moveGripper(200, GripperDirection.CLOSE, 20)
            time.sleep(1)
            motor.moveGripper(10000, GripperDirection.OPEN, 30)
            time.sleep(10)

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

