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

def sigint_handler(sig, frame):
    del actions
    del motor

    GPIO.cleanup()
    sys.exit(0)

class CubertActions:

    _defaul_move_speed = 40

    def __init__(self, motor:Motor.CubertMotor,  vision:Vision.CubertVision, solver:Solver.Solver, default_move_speed=10, calibrate_distance=False):
        self.motor = motor
        self._defaul_move_speed = default_move_speed
        self.vision = vision
        self.solver = solver

        motor.enable()

        self.motor.home(calibrate_distance)

    def preformMove(self, move:CubertNotation, rotation:Motor.BaseRotation, move_speed=_defaul_move_speed):

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

        for i in range(6):
            
            img = self.vision.getImage()

            if i == 0 or i == 1 or i == 2:
                self.flip()
            elif i == 3:
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
                self.flip()
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)
            elif i == 4:
                self.flip()
                self.flip()

            print('------------------------------------------- \n' + img.shape)
            print('------------------------------------------- \n' + combinedImage.shape)

            combinedImage[0:height, i*height:(i+1)*height, 0:3] = img
            combinedMask[0:height, i*height:(i+1)*height, 0:3] = self.vision.mask
            
            if(writeConsole):
                print("Cube rotated, waiting 2 seconds for camera to stabilize")

            if(i != 5):
                time.sleep(2)

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
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW)

            elif move == 'Y':
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)

            elif move == 'P':
                self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW)
                
            elif move == 'b':
                self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
                
            elif move == 'B':
                self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)
                
            elif move == 'b':
                self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CW)
                    
        print("Cube should be solved")

    def flip(self, move_speed=10):
        self.motor.moveGripperToPos(Motor.GripperPosition.BOTTOM, move_speed)
        self.motor.closeHand()
        self.motor.moveGripperToPos(Motor.GripperPosition.TOP, move_speed)
        self.motor.openHand()

    def rotateFace(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_defaul_move_speed):
        self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE, move_speed)
        self.motor.closeHand()
        self.motor.spinBase(rotation, direction, move_speed, degrees_to_correct=15)
        self.motor.openHand()

    def rotateCube(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_defaul_move_speed):
        self.motor.spinBase(rotation, direction, move_speed)

    def zen(self, move_speed=10):
        while True:
            move = random.randint(0,11)

            if random.randint(0,1):
                rotation = Motor.BaseRotation.QUARTER
            else:
                rotation = Motor.BaseRotation.HALF

            self.preformMove(move, rotation, move_speed)
  
            # if move == 2:
            #     self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            # elif move == 3:
            #     self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            # elif move == 4:
            #     self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            # elif move == 5:
            #     self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW, move_speed)
            # elif move == 6:
            #     self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed)
            # elif move == 7:
            #     self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed)
            # elif move == 8:
            #     self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed)
            # elif move == 9:
            #     self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CW, move_speed)
            # else:
            #     self.flip()

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

