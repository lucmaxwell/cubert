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

# from RubikCubeEnv import RubiksCubeEnv


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

    _default_base_speed = 400
    _base_accel_frac    = 0.05      # Base Max Speed Point 
    _default_arm_speed  = 200
    _arm_accel_frac     = 0.15      # Arm Max Speed Point
    _cube_face_spun     = False     # tracks if cube state was spun recently

    _grip_delay = 0.001
    _apex_delay = 0.001

    def __init__(self, motor:Motor.CubertMotor,  vision:Vision.CubertVision, solver:Solver.Solver, calibrate_distance=False, resize_cubelets=True):
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
        # self._defaul_move_speed = default_move_speed

        # enable and home motors
        motor.enable()

        self.motor.home(calibrate_distance)
        if resize_cubelets: self.sizeCubelet()

    def preformMove(self, move:CubertNotation, rotation:Motor.BaseRotation, move_speed=_default_base_speed, acceleration=True):
        """
        Purpose: Preform a given move on the Rubik's cube

        Inputs:
            - move:         CubertNotation of move to make
            - rotation:     BaseRotation to specify amount to rotate
            - move_speed:   Speed to preform movement
        """

        if move == CubertNotation.BACK_CCW:
            self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed, acceleration)

        elif move == CubertNotation.BACK_CW:
            self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CCW, move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed, acceleration)

        elif move == CubertNotation.BOTTOM_CCW:
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed, acceleration)

        elif move == CubertNotation.BOTTOM_CW:
            self.rotateFace(rotation, Motor.Direction.CW, move_speed, acceleration)

        elif move == CubertNotation.FRONT_CCW:
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed, acceleration)

        elif move == CubertNotation.BOTTOM_CW:
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed, acceleration)

        elif move == CubertNotation.LEFT_CCW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed, acceleration)
            
        elif move == CubertNotation.LEFT_CW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed, acceleration)

        elif move == CubertNotation.RIGHT_CCW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed, acceleration)
            
        elif move == CubertNotation.RIGHT_CW:
            self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed, acceleration)

        elif move == CubertNotation.TOP_CCW:
            self.flip(move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CCW, move_speed, acceleration)

        elif move == CubertNotation.TOP_CW:
            self.flip(move_speed, acceleration)
            self.flip(move_speed, acceleration)
            self.rotateFace(rotation, Motor.Direction.CW, move_speed, acceleration)

    def getAllImages(self, writeConsole=False):
        combinedShape = list(self.vision.lowerResolution) + [3]
        combinedShape[1] = combinedShape[1] * 6
        combinedShape = tuple(combinedShape)
        height = combinedShape[0]

        combinedImage = np.zeros(combinedShape, np.uint8)
        combinedMask = np.zeros(combinedShape, np.uint8)

        self.motor.homeLight()

        for i in range(6):
            
            self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE_CUBE)
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

    def ai_solve(self):
        # print("Taking images")
        # cube, mask = self.getAllImages(True)
        # print("Images taken")
        # print()
        #
        # # Find cube state
        # print("Finding cube state")
        # cubeState, outImage = self.vision.getCubeState(cube, mask, 3, 18, True)
        # print("Got cube state")
        # print(cubeState)
        # print()
        #
        # # Create the environment
        # env = RubiksCubeEnv()
        #
        # # Create the AI model
        # policy_kwargs = dict(
        #     features_extractor_class=network_configuration,
        #     features_extractor_kwargs=dict(features_dim=env.action_space.n)
        # )
        # training_model = DQN.load(model_file_path,
        #                           env=env,
        #                           verbose=2,
        #                           device="cuda")
        #
        # # Find cube solution
        # # Attempt to solve three times
        # print("Finding solution")
        # obs = env.set_observation(cubeState)
        # solved = False
        # count = 0
        # action_list = []
        # while not solved and count < 3:
        #     count += 1
        #
        #     # Determine action and take step
        #     action, _ = model.predict(obs, deterministic=True)
        #     obs, _, done, _, _ = env.step(action)
        #
        # # Check if the cube has been solved
        # done = env.is_solved()
        #
        #
        #
        # print("Found solution")
        # print(solution)
        # print()
        #
        # # Abort if the solver had an error
        # if (solution.startswith("Error: ")):
        #     print("Aborting solution attempt")
        #     return
        #
        # # Translate solution
        # print("Translating to cubertish")
        # cubertSolution = self.solver.cubertify(solution)
        # print("Translated to cubertish")
        # print(cubertSolution)
        # print()
        #
        # # Send instructions
        # print("Sending instructions")
        # for move in cubertSolution:
        #     if move == 'X':
        #         self.flip(acceleration=True)
        #
        #     elif move == 'y':
        #         self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, acceleration=False)
        #
        #     elif move == 'Y':
        #         self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, acceleration=False)
        #
        #     elif move == 'P':
        #         self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW, acceleration=False)
        #
        #     elif move == 'b':
        #         self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, acceleration=False)
        #
        #     elif move == 'B':
        #         self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW, acceleration=False)
        #
        #     elif move == 'p':
        #         self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CW, acceleration=False)
        #
        # print("Cube should be solved")
        pass


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
                self.flip(acceleration=True)

            elif move == 'D':
                self.doubleFlip(acceleration=True)

            elif move == 'y':
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CW, acceleration=False)

            elif move == 'Y':
                self.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, acceleration=False)

            elif move == 'P':
                self.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW, acceleration=False)
                
            elif move == 'b':
                self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, acceleration=False)
                
            elif move == 'B':
                self.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW, acceleration=False)
                
            elif move == 'p':
                self.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CW, acceleration=False)
                    
        print("Cube should be solved")

    def sizeCubelet(self):
        print("Resizing Cubelets")
        self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE, 50)
        self.motor.closeHand()
        time.sleep(0.5)
        self.motor.resizeCubelet(self.vision.getCubletSize())
        self.motor.openHand()


    def flip(self, move_speed=_default_arm_speed, acceleration=True):
        """
        Purpose: Flip the Rubik's cube

        Inputs:
            - move_speed:   Speed to preform flip
            - acceleration: If True acceleration enabled
        """

        self.motor.moveGripperToPos(Motor.GripperPosition.BOTTOM, move_speed, acceleration=acceleration, accel_fraction=self._arm_accel_frac)
        self.motor.closeHand()
        self.motor.moveGripperToPos(Motor.GripperPosition.FLIP_TOP, move_speed, acceleration=acceleration, accel_fraction=self._arm_accel_frac)
        time.sleep(self._apex_delay)
        self.motor.moveGripperToPos(Motor.GripperPosition.DROPOFF, move_speed, acceleration=acceleration, accel_fraction=self._arm_accel_frac)
        self.motor.openHand()

    def doubleFlip(self, move_speed=_default_arm_speed, acceleration=True):
        """
        Purpose: Flip the Rubik's cube twice

        Inputs:
            - move_speed:   Speed to preform flip
            - acceleration: If True acceleration enabled
        """

        self.flip(move_speed, acceleration)

        if self._cube_face_spun:
            # the Noah manuever
            self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE_CUBE, move_speed, acceleration=acceleration, accel_fraction=self._arm_accel_frac)
            time.sleep(self._grip_delay)
            self.motor.closeHand()
            steps_ccw   = self.motor.moveBaseDegrees(30, Motor.Direction.CCW, move_speed)
            steps_cw    = self.motor.moveBaseDegrees(38, Motor.Direction.CW, move_speed)
            self.motor.moveBase(steps_cw - steps_ccw, Motor.Direction.CCW, move_speed)
            time.sleep(self._grip_delay)
            self.motor.openHand()

            self._cube_face_spun = False

        self.flip(move_speed, acceleration)

    def rotateFace(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_default_base_speed, acceleration=True):
        """
        Purpose: Rotate a face of the Rubik's cube in the given direction

        Inputs:
            - rotation:     BaseRotation to specify amount to rotate
            - direction:    Direction to rotate in
            - move_speed:   Speed to preform rotation
            - acceleration: If True acceleration enabled
        """

        self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE_CUBE, move_speed, acceleration=acceleration, accel_fraction=self._arm_accel_frac)
        time.sleep(self._grip_delay)
        self.motor.closeHand()
        self.motor.moveBaseSpin(rotation, direction, move_speed, degrees_to_correct=8, acceleration=acceleration, accel_fraction=self._base_accel_frac)
        time.sleep(self._grip_delay)
        self.motor.openHand()

        self._cube_face_spun = True

    def rotateCube(self, rotation:Motor.BaseRotation, direction:Motor.Direction, move_speed=_default_base_speed, acceleration=True):
        """
        Purpose: Rotate the entire Rubik's cube in the given direction

        Inputs:
            - rotation:     BaseRotation to specify amount to rotate
            - direction:    Direction to rotate in
            - move_speed:   Speed to preform rotation
            - acceleration: If True acceleration enabled
        """

        self.motor.moveBaseSpin(rotation, direction, move_speed, acceleration=acceleration, accel_fraction=self._base_accel_frac)

    def scramble(self, num_moves):
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

            self.preformMove(move, rotation)


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

    
def sigint_handler(sig, frame):
    del actions
    del motor

    GPIO.cleanup()
    sys.exit(0)

def test_flip_speed(actions:CubertActions):
    for i in range(10,400):
        print("\n\n\n\nCurrent Speed: %d\n\n\n\n" % (i))
        actions._cube_face_spun = True
        # actions.doubleFlip(i)
        actions.flip(i)
        # # the Noah manuever
        actions.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE_CUBE, 400, acceleration=True, accel_fraction=actions._arm_accel_frac)
        actions.motor.closeHand()
        steps_ccw = actions.motor.moveBaseDegrees(30, Motor.Direction.CCW, 400)
        steps_cw = actions.motor.moveBaseDegrees(38, Motor.Direction.CW, 400)
        actions.motor.moveBase(steps_cw - steps_ccw, Motor.Direction.CCW, 400)
        actions.motor.openHand()
        

def test_spin_speed(actions:CubertActions):
    for i in range(5,200):
        print("\n\n\n\nCurrent Speed: %d\n\n\n\n" % (2*i))
        actions.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW, move_speed=2*i)
        print("\n\n\n\nCurrent Speed: %d\n\n\n\n" % (2*i+1))
        actions.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW, move_speed=2*i+1)

def test_spin(actions:CubertActions):

    print("\n\nTesting Spin\n----------------------")
    print("1: Quarter Spin Clockwise")
    print("2: Quarter Spin Counter-Clockwise")
    print("3: Half Spin Clockwise")
    print("4: Half Spin Counter-Clockwise")
    print("5: Full Spin Clockwise")
    print("6: Full Spin Counter-Clockwise")
    value = input()

    if 1 == int(value):
        print("\nStress Test Quarter Spin Clocwise")
        for i in range(200):
            actions.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
            print(i+1)

    elif 2 == int(value):
        print("\nStress Test Quarter Spin Counter-Clocwise")
        for i in range(200):
            actions.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)
            print(i+1)

    elif 3 == int(value):
        print("\nStress Test Half Spin Clocwise")
        for i in range(200):
            actions.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CW)
            print(i+1)

    elif 4 == int(value):
        print("\nStress Test Half Spin Counter-Clocwise")
        for i in range(200):
            actions.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CCW)
            print(i+1)

    elif 5 == int(value):
        print("\nStress Test Full Spin Clocwise")
        for i in range(200):
            actions.rotateFace(Motor.BaseRotation.FULL, Motor.Direction.CW)
            print(i+1)

    elif 6 == int(value):
        print("\nStress Test Full Spin Counter-Clocwise")
        for i in range(200):
            actions.rotateFace(Motor.BaseRotation.FULL, Motor.Direction.CCW)
            print(i+1)

    else:
        print("\nERROR: Value out of Range!")

def test_flip(actions:CubertActions):
    print("\n\nTesting Fli[\n----------------------")
    print("1: Single Flip")
    print("2: Double Flip w/ Error")

    value = input()

    if 1 == int(value):
        print("\nTesting Single Flip")
        for i in range(200):
            actions.flip()
            print(i+1)

    elif 2 == int(value):
        print("\nTesting Double Flip w/ Error")
        for i in range(200):
            actions._cube_face_spun = True # Set error correction flag
            # actions.doubleFlip()
            actions.flip()
            # the Noah manuever
            actions.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE_CUBE, 400, acceleration=True, accel_fraction=actions._arm_accel_frac)
            actions.motor.closeHand()
            steps_ccw = actions.motor.moveBaseDegrees(30, Motor.Direction.CCW, 400)
            steps_cw = actions.motor.moveBaseDegrees(38, Motor.Direction.CW, 400)
            actions.motor.moveBase(steps_cw - steps_ccw, Motor.Direction.CCW, 400)
            actions.motor.openHand()
            print(i+1)

    else:
        print("\nERROR: Value out of Range!")
    

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
    vision = Vision.CubertVision()
    solver = Solver.Solver()
    motor = Motor.CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin, current_sensor)

    actions = CubertActions(motor,vision,solver)

    print("Options:")
    print("0:\tStress Test Flip")
    print("1:\tStress Test Spin")
    print("2:\tStress Test Flip Speed")
    print("3:\tStress Test Spin Speed")
    value = input()

    if 0 == int(value):
        test_flip(actions)
    elif 1 == int(value):
        test_spin(actions)
    elif 2 == int(value):
        test_flip_speed(actions)
    elif 3 == int(value):
        test_spin_speed(actions)

    del actions
    del motor

    GPIO.cleanup()

