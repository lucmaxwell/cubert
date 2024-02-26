import RPi.GPIO as GPIO
import signal
import sys
import time
from enum import Enum, IntEnum
from TMC2209MotorLib.src.TMC_2209.TMC_2209_StepperDriver import *
import ctypes
import CurrentSensor
import statistics

# Parameter
MAX_SPEED = 3.3 # DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING.
MIN_SPEED = 0.000001

# library contains usleep
libc = ctypes.CDLL('libc.so.6')

# Define enumerations
class Direction(IntEnum):
    """Determines direction of motor spin"""
    CCW = 0
    CW  = 1

class GripperDirection(Enum):
    """Dicatets the way to move the gripper assembly"""
    UP      = 0
    DOWN    = 1
    OPEN    = 2
    CLOSE   = 3

class MotorType(IntEnum):
    """Identifies the motor to access"""
    BASE    = 0
    LEFT    = 1
    RIGHT   = 2

class GripperPosition(Enum):
    """Used in determining and labeling current gripper position in gauntry"""
    UNKNOWN         = 0
    TOP_ENDSTOP     = 1
    BOTTOM_ENDSTOP  = 2
    MIDDLE          = 3
    MIDDLE_CUBE     = 4
    TOP             = 5
    BOTTOM          = 6
    DROPOFF         = 7
    PICKUP          = 8
    FLIP_TOP        = 9

class HandState(Enum):
    """Used in determining and labeling current gripper hand state"""
    UNKOWN      = 0
    CLOSED      = 1
    OPEN        = 2
    OPEN_MAX    = 3

class BaseRotation(Enum):
    """Dictates amount of rotation"""
    QUARTER = 0
    HALF    = 1
    FULL    = 2

# ported vebatum from ESP code
def get_step_delay(velocity):
    """
    Purpose: detemine delay between steps for desired speed

    Inputs:
        - velocity: percentage of maximum speed between 0 and 200
    
    Output:
        - Integer number of microseconds to delay
    """
    v = max(min(velocity, 400), 1)
    x = MIN_SPEED + v * (MAX_SPEED - MIN_SPEED) / 100
    delay_duration = (1 / (0.0003 * x)) / 10
    return round(delay_duration)

def get_motor_velocity(move_speed, speed_up_fraction, curr_steps, total_steps):
    point1 = total_steps * speed_up_fraction
    point2 = total_steps * (1 - speed_up_fraction)
    if curr_steps < point1:
        vel = move_speed * curr_steps / point1
    elif curr_steps > point2:
        vel = move_speed * (total_steps - curr_steps) / point1
    elif curr_steps <= point2 and curr_steps >= point1:
        vel = move_speed

    return vel

class CubertMotor:

    # class constants
    _USE_UART = False # DON'T USE UART VERY BROKEN RIGHT NOW!!!

    _ACTUAL_STEPS   = 200               # number of steps in motor
    _MICROSTEPS     = 8                 # set microstep resolution
    _GEAR_RATIO     = 12                 # cube base gear ratio

    _MAX_CURRENT    = 700               # max current draw of motors in mA

    _DISTANCE_AT_BOTTOM     = 14.20     # distance from base to gripper when at bottom position in mm
    _DISTANCE_AT_TOP        = 64.22     # distance from base to gripper when at top position in mm

    _ENDSTOP_OFFSET_GAUNTRY = 2         # number of mm to stop at to avoid hitting top and bottom endstops
    _ENDSTOP_OFFSET_GRIPPER = 35        # number of steps to stop at to avoid hitting gripper endstop

    _DEFAULT_MOVE_SPEED     = 50        # default speed to preform moves at
    _DEFAULT_SPEED_UP_FRAC  = 0.10      # default point at which max speed is reached

    _TOLERANCE              = 1         # Tolerance in steps for determining gripper location
    _SANITY_STEP            = 100       # Tolerance for sanity checking _steps_from_bottom

    # derived class constants
    _STEPS_PER_REV      = _ACTUAL_STEPS * _MICROSTEPS   # number of steps per revolution of motor shaft
    _STEPS_PER_BASE_REV = _STEPS_PER_REV * _GEAR_RATIO  # number of steps per revolution of cube base

    _DISTANCE_FROM_BOTTOM_TO_TOP    = _DISTANCE_AT_TOP - _DISTANCE_AT_BOTTOM    # distance from gripper travel bottom to top in mm

    # determines if enstops are pressed
    _top_endstop_pressed        = False
    _bottom_endstop_pressed     = False
    _gripper_endstop_pressed    = False

    # class variables
    _steps_per_mm           = -1                        # number of steps to move gripper 1mm
    _steps_from_bottom      = -1                        # current number of step to get to the bottom endstop
    _steps_total_travel     = -2                        # number to steps to travel from enstop to endstop in gauntry
    _steps_to_close         = 345                       # number of steps until gripper is considered closed

    _cubelet_size           = 19                        # cublet size in mm

    _current_gripper_pos    = GripperPosition.UNKNOWN   # tracks the current gripper state
    _current_hand_state     = HandState.UNKOWN          # tracks the current gripper hand state

    _motor_dir_base         = Direction.CCW             # tracks base motor direction
    _motor_dir_left         = Direction.CCW             # tracks left motor direction
    _motor_dir_right        = Direction.CCW             # tracks right motor direction

    _gripper_homed          = False                     # False if gripper requires homing
    _base_homed             = False                     # False if base requires homing

    # derive class variables
    _dropoff_height         = _DISTANCE_AT_BOTTOM + 1.75 * _cubelet_size           # height in mm to release cube at
    _cube_middle_height     = _DISTANCE_AT_BOTTOM + 0.9 * _cubelet_size         # height of cube center
    _flip_apex_height       = _DISTANCE_AT_BOTTOM + 2.3 * _cubelet_size         # highest point when flipping cube
    _pickup_height          = _DISTANCE_AT_BOTTOM + _cubelet_size / 30           # height to grab cube at


    def __init__(self, enable_pin, step_pin_list, dir_pin_list, top_end_pin, bottom_end_pin, grip_end_pin, current_sensor:CurrentSensor.CubertCurrentSensor):
        """
        Purpose: Initialiser function

        Inputs:
            - enable_pin:       the pin number which enables all stepper motors
            - step_pin_list:    list of step pins for base, left and right motors (in that order)
            - dir_pin_list:     list of dir pins for base, left and right motors (in that order)
            - top_end_pin:      the pin number connected to the top endstop
            - bottom_end_pin:   the pin number connected to the bottom endstop
            - grip_end_pin:     the pin number connected to the gripper endstop
            - current_sensor:   CubertCurrentSensor object for measuring currents
        """

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

        # store current sensor
        self._current_sensor = current_sensor

        # setup motors
        for tmc in self.tmc_list:
            tmc.set_current(700)
            tmc.set_microstepping_resolution(self._MICROSTEPS)
            tmc.set_interpolation(True)
            tmc.set_direction_pin(Direction.CCW)

        # store enstop pins
        self._top_end_pin       = top_end_pin
        self._bottom_end_pin    = bottom_end_pin
        self._grip_end_pin      = grip_end_pin

        # setup enstops
        GPIO.setup(   top_end_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(bottom_end_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(  grip_end_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # validate endstop state
        if not GPIO.input(top_end_pin)      : self._top_endstop_pressed = True
        if not GPIO.input(bottom_end_pin)   : self._bottom_endstop_pressed = True
        if not GPIO.input(grip_end_pin)     : self._gripper_endstop_pressed = True

        # setup endstop interrupts
        GPIO.add_event_detect(top_end_pin, GPIO.BOTH,
                              callback=self.top_endstop_callback)
        GPIO.add_event_detect(bottom_end_pin, GPIO.BOTH,
                              callback=self.bottom_endstop_callback)
        GPIO.add_event_detect(grip_end_pin, GPIO.BOTH,
                              callback=self.gripper_endstop_callback)
        
        print("Motor Initialized")



    def __del__(self):
        """
        Purpose: Clean up resources
        """
        print("Deleting Motor")

        # disable stepper motors
        self.disable()

        # remove tmc objects
        for tmc in self.tmc_list:
            del(tmc)

        # cleanup gpio pins
        GPIO.cleanup()



    def enable(self):
        """
        Purpose: Enable stepper motors
        """
        self.tmc_base.set_motor_enabled(True)

    def disable(self):
        """
        Purpose: Disable stepper motors
        """
        self.tmc_base.set_motor_enabled(False)

    def stop(self):
        """
        Purpose: Stop all stepper motors from running

        ***NOTE: ONLY WORKS WITH UART***
        """
        for tmc in self.tmc_list:
            tmc.stop()


    # define calibration functions
    def home(self, calibrate_distance=False):
        """
        Purpose: Home all motors

        Inputs:
            - calibrate_distance: Boolean to determine if distance should be changed from defaults
        """
        print("Begining Homing")

        if calibrate_distance: self.calibrateDistance()

        # move gripper away from base
        self.openHand()
        self.moveGripperToPos(GripperPosition.TOP_ENDSTOP)

        # home components
        self.homeBase()
        self.homeGripper()
        # self.calibrateGripStrength()

        print("Homing Finished")

    def homeGripper(self):
        """
        Purpose: Home the gripper and determine steps to travers gauntry
        """
        print("Homing Gripper")

        self.openHand()

        # determine steps to traverse gauntry
        self.moveGripperToPos(GripperPosition.BOTTOM_ENDSTOP, 20)
        self._steps_total_travel = self.moveGripperToPos(GripperPosition.TOP_ENDSTOP, 20)

        # update gripper position in steps
        self.changeRelativeLocation(0,None)

        # center gripper
        self.moveGripperToPos(GripperPosition.MIDDLE)

        # determine steps require per mm
        self._steps_per_mm = self._steps_total_travel / self._DISTANCE_FROM_BOTTOM_TO_TOP

        self._gripper_homed = True

    def homeBase(self):
        """
        Purpose: Home the gripper and determine steps to travers gauntry
        """
        print("Homing Base")

        # setup function variables
        step_delay = 0
        short_delay = 0
        long_delay  = 600

        queue_length = 145
        queue = []
        median = -1
        divisor = 1

        direction = Direction.CCW

        threshold = 100

        times_crossed = 0

        # if suspected in transfer coil get away from coil
        if self._current_sensor.getChannelCurrent(CurrentSensor.CurrentChannel.BASE_LIGHT) > threshold:
            self.moveBaseSpin(BaseRotation.QUARTER, Direction.CW)

        # fill queue
        for i in range(queue_length):
            queue.append(self._current_sensor.getChannelCurrent(CurrentSensor.CurrentChannel.BASE_LIGHT))
            self.stepBase(direction, short_delay)
            libc.usleep(short_delay)

        median = statistics.median(queue)

        # find transfer coil
        while times_crossed < 40:          

            self.stepBase(direction, step_delay)

            # cycle queue
            queue.append(self._current_sensor.getChannelCurrent(CurrentSensor.CurrentChannel.BASE_LIGHT))
            queue.pop(0)

            median = statistics.median(queue)

            # check if fine scanning required
            if median > 105:
                step_delay = long_delay

            else:
                step_delay = short_delay

            # detect if in transfer coil
            if divisor % 7 == 0:
                if median >= threshold:
                    times_crossed += 1
                elif median < 90:
                    if times_crossed > 0:
                        print("Times Crossed: %d" % times_crossed)
                    times_crossed = 0
                    step_delay - short_delay

                divisor = 1

            else:
                divisor += 1

            libc.usleep(step_delay)

        # settle in nearest real step
        self.disable()
        time.sleep(0.1)
        self.enable()

        self._base_homed = True

    def homeLight(self):
        """
        Purpose: Find the transfer coil and turn the light on
        """
        
        if self._base_homed:
            queue = []
            threshold = 100
            light_found = False
            attempts = 0

            while attempts < 4:
                for i in range(10):
                    queue.append(self._current_sensor.getChannelCurrent(CurrentSensor.CurrentChannel.BASE_LIGHT))

                if statistics.median(queue) > threshold:
                    return
                
                else:
                    queue.clear()
                    self.moveBaseSpin(BaseRotation.QUARTER, Direction.CCW)

        
        self.homeBase()

    def calibrateGripStrength(self):
        """
        Purpose: Determine a acceptable grip strength for grabbiing the cube
        """

        start_time = time.time()

        warmup_time = 0.6

        step_delay = get_step_delay(10)
        steps_done = 0

        self.moveGripperToPos(GripperPosition.MIDDLE, 50)

        self._current_sensor.clearSkipFlag()

        while (steps_done < 750 and not sensor.getMotorSkipped()) or time_elapsed < warmup_time:
            self.stepGripper(GripperDirection.CLOSE, step_delay)
            libc.usleep(step_delay)
            steps_done += 1
            time_elapsed = time.time() - start_time

        self._steps_to_close = steps_done + 3

        print(self._steps_to_close)


    def calibrateDistance(self):
        """
        Purpose: Change system defaults to properly reflect gauntry dimensions
        """
        print("Calibrating Distance")

        self.moveGripperToPos(GripperPosition.BOTTOM_ENDSTOP, 50, True)

        dist = input("Input Distance Measured Between Gripper and Base: ")
        self._DISTANCE_AT_BOTTOM = float(dist)

        self.moveGripperToPos(GripperPosition.TOP_ENDSTOP, 50, True)

        dist = input("Input Distance Measured Between Gripper and Base: ")
        self._DISTANCE_AT_TOP = float(dist)

    def resizeCubelet(self, cubelet_size):
        self._cubelet_size = cubelet_size

        _dropoff_height         = self._DISTANCE_AT_BOTTOM + 1.75 * self._cubelet_size           # height in mm to release cube at
        _cube_middle_height     = self._DISTANCE_AT_BOTTOM + 0.9 * self._cubelet_size         # height of cube center
        _flip_apex_height       = self._DISTANCE_AT_BOTTOM + 2.3 * self._cubelet_size         # highest point when flipping cube
        _pickup_height          = self._DISTANCE_AT_BOTTOM + self._cubelet_size / 30


    # define callback functions
    def top_endstop_callback(self, channel):
        """
        Purpose: Callback for when top enstop hit or released
        """
        if not GPIO.input(self._top_end_pin) and not self._top_endstop_pressed:
            self._top_endstop_pressed = True
            print("Top Endstop Pressed")

        elif GPIO.input(self._top_end_pin) and self._top_endstop_pressed:
            self._top_endstop_pressed = False
            print("Top Endstop Released")

    def bottom_endstop_callback(self, channel):
        """
        Purpose: Callback for when bottom enstop hit or released
        """
        if not GPIO.input(self._bottom_end_pin) and not self._bottom_endstop_pressed:
            self._bottom_endstop_pressed = True
            print("Bottom Endstop Pressed")

        elif GPIO.input(self._bottom_end_pin) and self._bottom_endstop_pressed:
            self._bottom_endstop_pressed = False
            print("Bottom Endstop Released")

    def gripper_endstop_callback(self, channel):
        """
        Purpose: Callback for when gripper enstop hit or released
        """
        if not GPIO.input(self._grip_end_pin) and not self._gripper_endstop_pressed:
            self._gripper_endstop_pressed = True
            self._current_hand_state = HandState.OPEN_MAX
            print("Gripper Endstop Pressed")

        elif GPIO.input(self._grip_end_pin) and self._gripper_endstop_pressed:
            self._gripper_endstop_pressed = False
            print("Gripper Endstop Released")


    # define pseudo-pointer functions
    def get_top_endstop_pressed(self):
        """
        Purpose: Pseudo-pointer for _top_enstop_pressed variable

        Output:
            - Boolean: self._top_endstop_pressed
        """
        return self._top_endstop_pressed
    
    def get_bottom_endstop_pressed(self):
        """
        Purpose: Pseudo-pointer for _bottom_enstop_pressed variable

        Output:
            - Boolean: self._bottom_endstop_pressed
        """
        return self._bottom_endstop_pressed
    
    def get_gripper_endstop_pressed(self):
        """
        Purpose: Pseudo-pointer for _gripper_enstop_pressed variable

        Output:
            - Boolean: self._gripper_endstop_pressed
        """
        return self._gripper_endstop_pressed

    def return_false(self):
        """
        Purpose: Required because of other Pseudo-pointers

        Output:
            - Boolean: False
        """
        return False



    # depreciated spin base function
    def spinBaseDep(self, degrees_to_rotate, move_direction, move_speed, degrees_to_correct=0, acceleration=0):
        """JUST DON"T USE!!!"""
        revolutions = _GEAR_RATIO * degrees_to_rotate  / 360.0
        correction  = _GEAR_RATIO * degrees_to_correct / 360.0

        if move_direction == Direction.CCW:
            revolutions *= -1
            correction  *= -1

        self.tmc_base.set_vactual_rpm(move_speed, revolutions=(revolutions+correction), acceleration=acceleration)

        if abs(correction) > 0:
            self.tmc_base.set_vactual_rpm(move_speed, revolutions=-1*correction, acceleration=acceleration)


    # define gripper movement functions
    def moveGripperToPos(self, position:GripperPosition, move_speed=_DEFAULT_MOVE_SPEED, acceleration=False, accel_fraction=_DEFAULT_SPEED_UP_FRAC):
        """
        Purpose: Move the gripper to a specified position in the gauntry

        Inputs:
            - position:         GripperPosition that the gripper should move to
            - move_speed:       Speed at which the gripper should move at
            - acceleration:     If True acceleration is enabled
            - accel_fraction:   Point at which gripper hits max speed

        Output:
            - number of steps completed
        """
        
        endstop_to_check = self.return_false    # set to false
        steps            = 4000                 # a reasonably large number of steps
        direction        = None

        # check if gripper needs to move
        if self._current_gripper_pos == position:
            print("Gripper Already at Position")
            return 0

        # tracks steps completed
        steps_done = 0

        # calculate step delay
        step_delay = get_step_delay(move_speed)

        # check direction to step and which enstop to check if required
        if position == GripperPosition.TOP_ENDSTOP:
            print("Moving Gripper to Top Endstop")

            endstop_to_check = self.get_top_endstop_pressed

            direction = GripperDirection.UP

        elif position == GripperPosition.BOTTOM_ENDSTOP:
            print("Moving Gripper to Bottom Endstop")

            endstop_to_check = self.get_bottom_endstop_pressed

            direction = GripperDirection.DOWN

        elif position == GripperPosition.TOP:
            print("Moving Gripper to Top")

            self.moveGripperAbsoluteMM(self._DISTANCE_AT_TOP - self._ENDSTOP_OFFSET_GAUNTRY, move_speed=move_speed, acceleration=acceleration, accel_fraction=accel_fraction)
            steps = 0

            # steps, direction = self.getStepsAndDirection(self._steps_total_travel - self._ENDSTOP_OFFSET_GAUNTRY)

            # if direction == GripperDirection.UP:
            #     endstop_to_check = self.get_top_endstop_pressed
            # elif direction == GripperDirection.DOWN:
                # endstop_to_check = self.get_bottom_endstop_pressed

        elif position == GripperPosition.BOTTOM:
            print("Moving Gripper to Bottom")

            self.moveGripperAbsoluteMM(self._DISTANCE_AT_BOTTOM + self._ENDSTOP_OFFSET_GAUNTRY, move_speed=move_speed, acceleration=acceleration, accel_fraction=accel_fraction)
            steps = 0

            # steps, direction = self.getStepsAndDirection(self._ENDSTOP_OFFSET_GAUNTRY)

            # if direction == GripperDirection.UP:
            #     endstop_to_check = self.get_top_endstop_pressed
            # elif direction == GripperDirection.DOWN:
                # endstop_to_check = self.get_bottom_endstop_pressed  

        elif position == GripperPosition.MIDDLE:
            print("Moving Gripper to Middle")

            self.moveGripperAbsoluteMM(self._DISTANCE_FROM_BOTTOM_TO_TOP/2, move_speed=move_speed, acceleration=acceleration, accel_fraction=accel_fraction)
            steps = 0

            # if self._current_gripper_pos == GripperPosition.TOP_ENDSTOP:
            #     endstop_to_check = self.get_bottom_endstop_pressed
            #     direction = GripperDirection.DOWN
            #     steps = self._steps_total_travel/2

            # elif self._current_gripper_pos == GripperPosition.BOTTOM_ENDSTOP:
            #     endstop_to_check = self.get_top_endstop_pressed
            #     direction = GripperDirection.UP
            #     steps = self._steps_total_travel/2

            # elif self._current_gripper_pos == GripperPosition.UNKNOWN or self._current_gripper_pos == GripperPosition.TOP or self._current_gripper_pos == GripperPosition.BOTTOM:

                # steps = self._steps_total_travel / 2 - self._steps_from_bottom

                # if steps > 0:
                #     endstop_to_check = self.get_top_endstop_pressed
                #     direction = GripperDirection.UP

                # elif steps < 0:
                #     endstop_to_check = self.get_bottom_endstop_pressed
                #     direction = GripperDirection.DOWN
                #     steps *= -1        

            # else:
            #     print("This Shouldn't Be Happening!")
            #     print(self._current_gripper_pos)

        elif position == GripperPosition.PICKUP:
            print("Moving to Pickup Point")

            self.moveGripperAbsoluteMM(self._pickup_height, move_speed=move_speed, acceleration=acceleration, accel_fraction=accel_fraction)
            steps = 0

        elif position == GripperPosition.FLIP_TOP:
            print("Moving to Flip Apex")

            self.moveGripperAbsoluteMM(self._flip_apex_height, move_speed=move_speed, acceleration=acceleration, accel_fraction=accel_fraction)
            steps = 0

        elif position == GripperPosition.MIDDLE_CUBE:
            print("Moving to Middle of Cube")

            self.moveGripperAbsoluteMM(self._cube_middle_height, move_speed=move_speed, acceleration=acceleration, accel_fraction=accel_fraction)
            steps = 0

        elif position == GripperPosition.DROPOFF:
            print("Moving to Dropoff Point")

            self.moveGripperAbsoluteMM(self._dropoff_height, move_speed=move_speed, acceleration=acceleration, accel_fraction=accel_fraction)
            steps = 0

        # step motors
        while (not endstop_to_check()) and steps_done < steps:

            if acceleration:
                vel = get_motor_velocity(move_speed, accel_fraction, steps_done, steps)
                step_delay = get_step_delay(vel)

            self.stepGripper(direction, step_delay)
            steps_done += 1
            libc.usleep(step_delay)

        print("Movement Complete")

        # update gripper location
        if direction is not None:
            self.changeRelativeLocation(steps_done, direction)

        return steps_done
    
    def moveGripperRelativeMM(self, mm_to_move, move_speed=_DEFAULT_MOVE_SPEED, acceleration=False, accel_fraction=_DEFAULT_SPEED_UP_FRAC):
        """
        Purpose: Move the gripper a number of mm in specified direction

        Inputs:
            - mm_to_move:       Number of mm to move, negaive moves gripper down
            - move_speed:       Speed to move the gripper at
            - acceleration:     If True acceleration enabled
            - accel_fraction:   Point at which max speed is reached
        """
        
        # get steps to move
        steps = round(abs(mm_to_move) * self._steps_per_mm)

        # determine move direction
        if mm_to_move > 0:
            direction = GripperDirection.UP
        elif mm_to_move < 0:
            direction = GripperDirection.DOWN

        self.moveGripper(steps, direction, move_speed, acceleration, accel_fraction)

    def moveGripperAbsoluteMM(self, mm_to_move_to, move_speed=_DEFAULT_MOVE_SPEED, acceleration=False, accel_fraction=_DEFAULT_SPEED_UP_FRAC):
        """
        Purpose: Move the gripper ato specified distance from base

        Inputs:
            - mm_to_move_to:    mm from base to move to
            - move_speed:       Speed to move the gripper at
            - acceleration:     If True acceleration enabled
            - accel_fraction:   Point at which max speed is reached
        """
        if mm_to_move_to > self._DISTANCE_AT_TOP:
            print("DISTANCE ABOVE TOP ENDSTOP!")
        elif mm_to_move_to < self._DISTANCE_AT_BOTTOM:
            print("DISTANCE BELOW BOTTOM ENDSTOP!")
        else:
            print("\n\nMoving to %5.2fmm" % mm_to_move_to)
            mm_to_move = mm_to_move_to - self.getPositionInMM()
            self.moveGripperRelativeMM(mm_to_move, move_speed, acceleration, accel_fraction)

    def moveGripper(self, steps, direction:GripperDirection, move_speed=_DEFAULT_MOVE_SPEED, acceleration=False, accel_fraction=_DEFAULT_SPEED_UP_FRAC):
        """
        Purpose: move the gripper a specified number of spet in the given direction

        Inputs:
            - steps:            number of steps to preform
            - direction:        GripperDirection to move in
            - move_speed:       Speed to move gripper at
            - acceleration:     If True acceleration enabled
            - accel_fraction:   Point at which max speed is reached
        """
        
        # no enstop to check
        endstop_to_check = self.return_false

        # tracks steps completed
        steps_done = 0

        # calculate step delay
        step_delay = get_step_delay(move_speed)

        # check direction to step
        if direction == GripperDirection.UP:
            print("Moving Gripper Up")

            endstop_to_check = self.get_top_endstop_pressed

        elif direction == GripperDirection.DOWN:
            print("Moving Gripper Down")

            endstop_to_check = self.get_bottom_endstop_pressed
        
        elif direction == GripperDirection.OPEN:
            print("Opening Gripper")

            endstop_to_check = self.get_gripper_endstop_pressed
        
        elif direction == GripperDirection.CLOSE:
            print("Closing Gripper")

        # move gripper
        while (not endstop_to_check()) and steps_done < steps:

            if acceleration:
                vel = get_motor_velocity(move_speed, accel_fraction, steps_done, steps)
                step_delay = get_step_delay(vel)

            self.stepGripper(direction, step_delay)
            steps_done += 1
            libc.usleep(step_delay)

        # track location
        self.changeRelativeLocation(steps_done, direction)

        return steps_done


    # define state updating functions
    def changeGripperPosition(self):
        """
        Purpose: Sets _current_gripper_pos to proper state
        """
        if self._steps_from_bottom == 0:
            self._current_gripper_pos = GripperPosition.BOTTOM_ENDSTOP

        elif self._steps_from_bottom == self._steps_total_travel:
            self._current_gripper_pos = GripperPosition.TOP_ENDSTOP

        elif self.checkIfInTolerance(self._steps_from_bottom, self._steps_total_travel - self._ENDSTOP_OFFSET_GAUNTRY):
            self._current_gripper_pos = GripperPosition.TOP

        elif self.checkIfInTolerance(self._steps_from_bottom, self._ENDSTOP_OFFSET_GAUNTRY):
            self._current_gripper_pos = GripperPosition.BOTTOM

        elif self.checkIfInTolerance(self._steps_from_bottom, self._steps_total_travel/2):
            self._current_gripper_pos = GripperPosition.MIDDLE

        elif self._steps_from_bottom < -self._SANITY_STEP or self._steps_from_bottom > self._steps_total_travel + self._SANITY_STEP:
            GPIO.cleanup()
            sys.exit(-1)

        else:
            self._current_gripper_pos = GripperPosition.UNKNOWN

        print("Current Gripper Position is ", self._current_gripper_pos)

    def checkIfInTolerance(self, value, target):
        """
        Purpose: Used to check if variable is within acceptable range

        Inputs:
            - value: value to check
            - target: target value to be in range of
        """
        return value <= target + self._TOLERANCE and value >= target - self._TOLERANCE

    def changeRelativeLocation(self, steps, direction:GripperDirection):
        """
        Purpose: Change number of step to bottom to reflect position change

        Inputs:
            - steps: number of steps made
            - direction: GripperDirection of movement
        """

        # check endstops and movement direction
        if self._bottom_endstop_pressed:
            self._steps_from_bottom = 0

        elif self._top_endstop_pressed:
            self._steps_from_bottom = self._steps_total_travel

        else:
            if direction == GripperDirection.UP:
                self._steps_from_bottom += steps

            elif direction == GripperDirection.DOWN:
                self._steps_from_bottom -= steps
        
        # update gripper position
        self.changeGripperPosition()

        # debug info
        if self._gripper_homed:
            print("Steps from Bottom: %d" % self._steps_from_bottom)
            print("Max Steps to Travel: %d" % self._steps_total_travel)
            print("Gripper Position is %5.2fmm from Base" % (self.getPositionInMM()))

    def getPositionInMM(self):
        """
        Purpose: Determine the position in mm from base

        Output:
            - position in mm from base
        """
        return (self._steps_from_bottom / self._steps_per_mm) + self._DISTANCE_AT_BOTTOM

    def getStepsAndDirection(self, step_target_abs):
        """
        Purpose: Find the number of steps and direction to get gripper to desired position

        Inputs:
            - step_target_abs: The number of steps from the bottom to position gripper at

        Outputs:
            - Returns a tuple. First value is number of steps and second is direction to move
        """

        delta_steps = step_target_abs - self._steps_from_bottom

        if delta_steps < 0:
            direction = GripperDirection.DOWN
            delta_steps *= -1

        else:
            direction = GripperDirection.UP

        return delta_steps, direction



    # define gripper hand functions
    def closeHand(self):
        """
        Purpose: Close gripper fingers on cube
        """

        print("Closing Hand")

        # only close if hand state known to be open
        if self._current_hand_state == HandState.OPEN_MAX or self._current_hand_state == HandState.OPEN:
            self.moveGripper(self._steps_to_close, GripperDirection.CLOSE, 75)
            self._current_hand_state = HandState.CLOSED

    def openHand(self):
        """
        Purpose: Open gripper fingers on cube
        """
        print("Opening Hand")

        # move until enstop hit
        self.moveGripper(self._steps_to_close - self._ENDSTOP_OFFSET_GRIPPER, GripperDirection.OPEN, 75)

        self._current_hand_state = HandState.OPEN



    # define cubert base spinning functions
    def moveBaseSpin(self, rotation:BaseRotation, direction:Direction, move_speed=_DEFAULT_MOVE_SPEED, degrees_to_correct=0, acceleration=False, accel_fraction=_DEFAULT_SPEED_UP_FRAC):
        """
        Purpose: Spin cube base fraction of turn

        Inputs:
            - rotation:             BaseRotation specifying amount to turn
            - direction:            Direction to rotate cube base
            - move_speed:           Speed to preform rotation
            - degrees_to_correct:   Number of degrees to preform correction
            - acceleration:         If True acceleration enabled
            - accel_fraction:       Point at which max speed is reached 
        """

        # verify valid number of degrees
        if degrees_to_correct < 0:
            print("CANNOT CORRECT NEGATIVE DEGREES!")
            degrees_to_correct = 0

        # rotate base
        if rotation == BaseRotation.QUARTER:
            print("Base Rotating 90 Degrees")
            degrees = 90 + degrees_to_correct

        elif rotation == BaseRotation.HALF:
            print("Base Rotating 180 Degrees")
            degrees = 180 + degrees_to_correct

        elif rotation == BaseRotation.FULL:
            print("Base Rotating 360 Degrees")
            degrees = 360 + degrees_to_correct

        else:
            print("ERROR: Could not find specified rotation type!")

        # rotate base
        self.moveBaseDegrees(degrees, direction, move_speed, acceleration, accel_fraction)

        # correct if required
        if degrees_to_correct > 0:
            
            print("Correcting Base %2.0f Degrees" % (degrees_to_correct))

            # reverse direction
            if direction == Direction.CCW:
                correction_direction = Direction.CW
            else:
                correction_direction = Direction.CCW 

            self.moveBaseDegrees(degrees_to_correct, correction_direction, move_speed, acceleration, accel_fraction)

    def moveBaseDegrees(self, degrees_to_rotate, direction:Direction, move_speed=_DEFAULT_MOVE_SPEED, acceleration=False, accel_fraction=_DEFAULT_SPEED_UP_FRAC):
        """
        Purpose: Rotate the cube base a specified number of degrees

        Inputs:
            - degrees_to_rotate:    Number of degrees to rotate
            - direction:            Direction to rotate cube base
            - move_speed:           Speed to preform rotation
            - acceleration:         If True acceleration enabled
            - accel_fraction:       Point at which max speed is reached 
        """
        
        # convert degrees to steps
        steps = round(self._STEPS_PER_BASE_REV * degrees_to_rotate / 360)

        self.moveBase(steps, direction, move_speed, acceleration, accel_fraction)

    def moveBase(self, steps, direction:Direction, move_speed=_DEFAULT_MOVE_SPEED, acceleration=False, accel_fraction=_DEFAULT_SPEED_UP_FRAC):
        """
        Purpose: Rotate the cube base a specified number of steps

        Inputs:
            - steps:            Number of steps to preform
            - direction:        Direction to rotate cube base
            - move_speed:       Speed to preform rotation
            - acceleration:     If True acceleration enabled
            - accel_fraction:   Point at which max speed is reached 
        """

        # count steps completed
        steps_done = 0

        print("Move speed is %f" % move_speed)

        # calculate step delay
        step_delay = get_step_delay(move_speed)

        last_speed = 0
        curr_speed = 0
        peak_hit   = False

        # spin for given number of steps
        for _ in range(steps):

            if acceleration:
                vel = get_motor_velocity(move_speed, accel_fraction, steps_done, steps)
                step_delay = get_step_delay(vel)
                last_speed = curr_speed
                curr_speed = step_delay

            # if curr_speed - last_speed > 0 and not peak_hit:
            #     print(last_speed)
            #     peak_hit = True


            self.stepBase(direction, step_delay)
            steps_done += 1
            libc.usleep(step_delay)

        return steps_done


    # define other movement functions
    def move(self, steps, direction:Direction, motor:MotorType, move_speed=_DEFAULT_MOVE_SPEED):
        """
        Purpose: Rotate the given motor a specified number of steps

        Inputs:
            - steps:            Number of steps to preform
            - direction:        Direction to rotate cube base
            - motor:            MotorType to actuate
            - move_speed:       Speed to preform rotation
        """

        # Calculate the delay time of the pulse
        stepDelay = get_step_delay(move_speed)

        # Spin with given number of steps
        for _ in range(steps):
            self.step(direction, motor)
            libc.usleep(stepDelay)



    def stepGripper(self, direction:GripperDirection, step_delay):
        """
        Purpose: Move gripper one step in specified direction

        Inputs:
            - direction:    GripperDirection to move in
            - delay:        Number of microseconds to delay
        """

        endstop_to_check = False

        # check direction to step
        if direction == GripperDirection.UP:
            left_dir = Direction.CCW
            right_dir = Direction.CW

            endstop_to_check = self._top_endstop_pressed

        elif direction == GripperDirection.DOWN:
            left_dir = Direction.CW
            right_dir = Direction.CCW

            endstop_to_check = self._bottom_endstop_pressed
        
        elif direction == GripperDirection.OPEN:
            left_dir = Direction.CCW
            right_dir = Direction.CCW

            endstop_to_check = self._gripper_endstop_pressed
        
        elif direction == GripperDirection.CLOSE:
            left_dir = Direction.CW
            right_dir = Direction.CW
            
        else:
            print("ERROR: Direction does not exist!")
            return -1

        # step gripper
        if not endstop_to_check:
            self.step(left_dir, MotorType.LEFT, step_delay)
            self.step(right_dir, MotorType.RIGHT, step_delay)

    def stepBase(self, direction:Direction, step_delay):
        """
        Purpose: Move cube base on step in given direction

        Inputs:
            - direction:    Direction to step in
            - step_delay:   Number of microseconds to delay
        """
        self.step(direction, MotorType.BASE, step_delay)

    def step(self, direction:Direction, motor:MotorType, step_delay):
        """
        Purpose: Move given motor one step in specified direction

        Inputs:
            - direction:    Direction to spin in
            - motor:        MotorType to rotate
            - step_delay:   Number of microseconds to delay
        """
        direction_changed = False

        if self._USE_UART:
            rev_per_sec = self._STEPS_PER_REV * step_delay / 1_000_000
            if direction == Direction.CCW: rev_per_sec *= -1
            self.tmc_list[motor].set_vactual_dur_alt(0, duration=step_delay)
            
        else:
            # set step direction if required
            if motor == MotorType.BASE and direction != self._motor_dir_base:
                self._motor_dir_base = not self._motor_dir_base
                direction_changed = True
            
            elif motor == MotorType.LEFT and direction != self._motor_dir_left:
                self._motor_dir_left = not self._motor_dir_left
                direction_changed = True

            elif motor == MotorType.RIGHT and direction != self._motor_dir_right:
                self._motor_dir_right = not self._motor_dir_right
                direction_changed = True
            

            if direction_changed:
                self.tmc_list[motor].set_direction_pin(direction)

            # step motor
            self.tmc_list[motor].make_a_step()
            # self._current_sensor.logCurrent(motor)

# testing functionality
def sigint_handler(sig, frame):
    global sensor
    global motor

    del motor
    del sensor

    GPIO.cleanup()
    sys.exit(0)

if __name__ == '__main__':
    motor_en_pin = 26
    motor_step_pin = [27, 6, 19]
    motor_dir_pin = [17, 5, 13]

    # End stop for arm
    end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
    end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
    end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

    # initialize motor
    sensor = CurrentSensor.CubertCurrentSensor()
    motor = CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin, sensor)

    signal.signal(signal.SIGINT, sigint_handler)

    # Spin
    print("Running motor...")
    try:
        motor.enable()

        motor.home()

        # motor.moveBase(round(19200/4), Direction.CCW, 75, True)

        # motor.moveBase(round(19200/4), Direction.CCW, 200, True)

        # motor.moveBaseDegrees(90, Direction.CCW, 200, True)

        # motor.moveGripperToPos(GripperPosition.BOTTOM_ENDSTOP, 50)
        # time.sleep(1)
        # motor.moveGripperToPos(GripperPosition.BOTTOM, 50)
        # time.sleep(1)
        # motor.moveGripperToPos(GripperPosition.PICKUP, 50)
        # time.sleep(1)
        # motor.moveGripperToPos(GripperPosition.MIDDLE_CUBE, 50)
        # time.sleep(1)
        motor.moveGripperToPos(GripperPosition.MIDDLE, 50)
        time.sleep(10)
        motor.calibrateGripStrength()
        time.sleep(20)
        # motor.moveGripperToPos(GripperPosition.DROPOFF, 50)
        # time.sleep(1)
        # motor.moveGripperToPos(GripperPosition.FLIP_TOP, 50)
        # time.sleep(1)
        # motor.moveGripperToPos(GripperPosition.TOP, 50)
        # time.sleep(1)
        # motor.moveGripperToPos(GripperPosition.TOP_ENDSTOP, 50)
        # time.sleep(1)

        print("Testing Complete!")

        motor.__del__()
        sensor.__del__()

        GPIO.cleanup()

        print("Cleaned Up!")

        # while True:
        #     # do nothing
        #     time.sleep(10)

        # print("Spinning CW 180")
        # motor.moveBaseSpin(180, MotorSpin.CLOCKWISE, 60)

        # print("Spinning CCW 180")
        # motor.moveBaseSpin(180, MotorSpin.COUNTER_CLOCKWISECLOCKWISE, 60)

        # print("Spinning CW 180 With Correction")
        # motor.moveBaseSpin(180, MotorSpin.CLOCKWISE, 60, 5)

        # print("Spinning CCW 180 With Correction")
        # motor.moveBaseSpin(180, MotorSpin.COUNTER_CLOCKWISE, 60, 5)

    except KeyboardInterrupt:
        pass
    # finally:
        # del motor

