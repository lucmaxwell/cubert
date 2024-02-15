import Motor
import time
import signal
import sys
import RPi.GPIO as GPIO
import random
from CurrentSensor import *
from enum import IntEnum
import Vision
import Solver
import numpy as np

class CubertNotation(IntEnum):
    """List of Face Spins"""
    BOTTOM_CW   = 0
    BOTTOM_CCW  = 1
    TOP_CW      = 2
    TOP_CCW     = 3
    FRONT_CW    = 4
    FRONT_CCW   = 5
    BACK_CW     = 6
    BACK_CCW    = 7
    LEFT_CW     = 8
    LEFT_CCW    = 9
    RIGHT_CW    = 10
    RIGHT_CCW   = 11

class CubertActions:

    _defaul_move_speed  = 75        # Default motor speed
    _cube_face_spun     = False     # tracks if cube state was spun recently

    def __init__(self, motor:Motor.CubertMotor,  vision:Vision.CubertVision, solver:Solver.Solver, default_move_speed=10, calibrate_distance=False):
        """
        Purpose: Setup CubertAction Class

        Inputs:
            - motor:                CubertMotor to enable actuation
            - vision:               CubertVision module to process images
            - solver:               Solver module to determine solution
            - default_move_speed:   To update default move speed
            - calibrate_distance:   If True begins distance calibration
        """
        
        # save object references
        self.motor = motor
        self.vision = vision
        self.solver = solver

        # change default move speed
        self._defaul_move_speed = default_move_speed

        # enable and home motors
        motor.enable()

        self.motor.home(calibrate_distance)

    def preformMove(self, move:CubertNotation, rotation:Motor.BaseRotation, move_speed=_defaul_move_speed):
        """
        Purpose: Preform a given move on the Rubik's cube

        Inputs:
            - move:         CubertNotation of move to make
            - rotation:     BaseRotation to specify amount to rotate
            - move_speed:   Speed to preform movement
        """

        if move == CubertNotation.BACK_CCW:
            self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.BACK_CW:
            self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.BOTTOM_CCW:
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.BOTTOM_CW:
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.FRONT_CCW:
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.BOTTOM_CW:
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.LEFT_CCW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)
            
        elif move == CubertNotation.LEFT_CW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.RIGHT_CCW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)
            
        elif move == CubertNotation.RIGHT_CW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

        elif move == CubertNotation.TOP_CCW:
            self.flip(move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed)

        elif move == CubertNotation.TOP_CW:
            self.flip(move_speed)
            self.flip(move_speed)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed)

    def getAllImages(self, writeConsole=False):
        combinedShape = list(self.vision.lowerResolution) + [3]
        combinedShape[1] = combinedShape[1] * 6
        combinedShape = tuple(combinedShape)
        height = combinedShape[0]

        combinedImage = np.zeros(combinedShape, np.uint8)
        combinedMask = np.zeros(combinedShape, np.uint8)

        self.motor.homeBase()

        for i in range(6):
            
            self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE)
            self.motor.closeHand()
            img = self.vision.getImage()
            self.motor.openHand()

            if i == 0 or i == 1 or i == 2:
                self.flip()
            elif i == 3:
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
                self.flip()
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)
            elif i == 4:
                self.flip()
                self.flip()

            combinedImage[0:height, i*height:(i+1)*height, 0:3] = img
            combinedMask[0:height, i*height:(i+1)*height, 0:3] = self.vision.mask
            
            if(writeConsole):
                print("Cube rotated, waiting 2 seconds for camera to stabilize")

            if(i != 5):
                time.sleep(0.5)

        return combinedImage, combinedMask

    def solve(self, writeImages=False):
        print(f"Starting cube solving sequence")

        # Take images
        print("Taking images")
        cube, mask = self.getAllImages(True)

        # Write output images
        if(writeImages):
            self.vision.writeImage("testingImage.png", cube)
            self.vision.writeImage("testingmask.png", mask)

        print("Images taken")
        print()
        
        # Find cube state
        print("Finding cube state")
        cubeState, outImage = self.vision.getCubeState(cube, mask, 3, 18, True)
        print("Got cube state")
        print(cubeState)
        print()

        # Find cube solution
        print("Finding solution")
        solution = self.solver.get3x3Solution(cubeState)
        print("Found solution")
        print(solution)
        print()

        # Abort if the solver had an error
        if(solution.startswith("Error: ")):
            print("Aborting solution attempt")
            return
        
        # Translate solution
        print("Translating to cubertish")
        cubertSolution = self.solver.cubertify(solution)
        print("Translated to cubertish")
        print(cubertSolution)
        print()

        # Send instructions
        print("Sending instructions")
        for move in cubertSolution:
            if move == 'X':
                self.flip()

            elif move == 'y':
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, 120)

            elif move == 'Y':
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, 120)

            elif move == 'P':
                self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW, 120)
                
            elif move == 'b':
                self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, 60)
                
            elif move == 'B':
                self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW, 60)
                
            elif move == 'p':
                self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CW, 60)
                    
        print("Cube should be solved")

    def flip(self, move_speed=_defaul_move_speed, acceleration=False):
        """
        Purpose: Flip the Rubik's cube

        Inputs:
            - move_speed:   Speed to preform flip
            - acceleration: If True acceleration enabled
        """

        self.motor.moveGripperToPos(Motor.GripperPosition.BOTTOM, move_speed, acceleration=acceleration)
        self.motor.closeHand()
        self.motor.moveGripperToPos(Motor.GripperPosition.TOP, move_speed, acceleration=acceleration)
        self.motor.openHand()

        if self._cube_face_spun:
            # the Noah manuever
            self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE, move_speed, acceleration=acceleration)
            self.motor.closeHand()
            self.motor.moveBaseDegrees(30, Motor.Direction.CCW)
            self.motor.moveBaseDegrees(30, Motor.Direction.CW)
            self.motor.openHand()

            self._cube_face_spun = False

    def rotateFace(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_defaul_move_speed, acceleration=False):
        """
        Purpose: Rotate a face of the Rubik's cube in the given direction

        Inputs:
            - rotation:     BaseRotation to specify amount to rotate
            - direction:    Direction to rotate in
            - move_speed:   Speed to preform rotation
            - acceleration: If True acceleration enabled
        """

        self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE, move_speed, acceleration=acceleration)
        self.motor.closeHand()
        self.motor.spinBase(rotation, direction, move_speed, degrees_to_correct=15, acceleration=acceleration)
        self.motor.openHand()

        self._cube_face_spun = True

    def rotateCube(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_defaul_move_speed, acceleration=False):
        """
        Purpose: Rotate the entire Rubik's cube in the given direction

        Inputs:
            - rotation:     BaseRotation to specify amount to rotate
            - direction:    Direction to rotate in
            - move_speed:   Speed to preform rotation
            - acceleration: If True acceleration enabled
        """

        self.motor.spinBase(rotation, direction, move_speed, acceleration=acceleration)

    def scramble(self, num_moves, move_speed=50):
        """
        Purpose: Scramble the Rubik's Cube

        Inputs:
            - num_moves:    Number of times to scramble
            - move_speed:   Speed to preform movements
        """
        for i in range(num_moves):
            move = random.randint(2,11)

            if random.randint(0,1):
                rotation = Motor.BaseRotation.QUARTER
            else:
                rotation = Motor.BaseRotation.HALF

            self.preformMove(move, rotation, move_speed)


    def zen(self, move_speed=10):
        """
        Purpose: Slowly preform random moves

        Inputs:
            - move_speed: For those who would like less inner peace with their zen
        """
        while True:
            move = random.randint(0,11)

            if random.randint(0,1):
                rotation = Motor.BaseRotation.QUARTER
            else:
                rotation = Motor.BaseRotation.HALF

            self.preformMove(move, rotation, move_speed)

    def enableLights(self, imageUrl, client, writeConsole=False):
        lights = np.zeros(4)

        # Find the orientation with the highest lightness
        for i in range(4):
            img = self.vision.getImage()
            average = np.average(img)
            lights[i] = average
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW)

            if(i < 4):
                time.sleep(0.25)

        # Spin to the lightest side
        spin = lights.argmax()
        if(spin == 1):
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
        elif(spin == 2):
            self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW)
        elif(spin == 3):
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)

    
def sigint_handler(sig, frame):
    del actions
    del motor

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

    signal.signal(signal.SIGINT, sigint_handler)

    current_sensor = CubertCurrentSensor()

    # initialize motor
    motor = Motor.CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin, current_sensor)

    actions = CubertActions(motor)

    time.sleep(2)

    actions.flip()
    time.sleep(1)
    actions.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)
    time.sleep(1)
    actions.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW)
    time.sleep(1)
    actions.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
    time.sleep(1)
    actions.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CCW)

    speed = input("Give Zen Mode Speed: ")

    actions.zen(int(speed))


    del actions
    del motor

    GPIO.cleanup()

