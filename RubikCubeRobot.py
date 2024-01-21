import time

import RPi.GPIO as GPIO

from RubikCube import RubikCube, Face
from Button import Button

from enum import Enum


class RobotAction(Enum):
    SPIN_BASE_CLOCKWISE = 0
    SPIN_BASE_COUNTERCLOCKWISE = 1
    GRIP_AND_FLIP = 2
    SPIN_CUBE_CLOCKWISE = 3
    SPIN_CUBE_COUNTERCLOCKWISE = 4


class RubiksCubeRobot:
    def __init__(self, BUTTON_PIN, LIGHT_PIN):
        self.LIGHT_PIN = LIGHT_PIN

        # Initialize the button
        self.button = Button(BUTTON_PIN, GPIO.LOW)



        # Initialize the cube's state
        self.cube = RubikCube(3)
        self.action_instruction_list = []

    def ai_solve(self):
        # Create the environment
        env = RubiksCubeEnv(num_scramble=num_scramble)

        # Scramble the cube
        original_obs = env.scramble(num_scramble)

        # Solve the puzzle
        # Allow multiple attempts
        done = False
        count = 0
        while count < 3 and not done:
            count += 1

            # Solve the cube
            obs = env.set_observation(original_obs)
            while not done:
                # Determine action and take step
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)

            # Check if the cube has been solved
            done = env.is_solved()

            if not multiple_attempts:
                break

        # Return
        return done

    def fake_ai_solve(self):
        pass

    def solve(self, cube_state_data):
        # Process the image to get the cube state
        cube_state_obs = self.processing_image(cube_state_data)

        # Initialize self.cube with the cube state
        self.cube.set_state_from_observation(cube_state_obs)

        # Reorient the cube to be the default? Is reorientation needed?
        # Save the spin instructions

        # Solve the cube and get the instructions
        solve_instruction_list = self.solve_cube()

        # Compute the instructions for the robot
        robot_instruction_list = self.get_robot_intructions(solve_instruction_list)

        # Validate that the instructions will solve the cube
        # Optimize the instruction, if the instructions undo each other, remove them

        # Return the instructions for the robot
        pass

    def processing_image(self, new_state):
        return None

    def solve_cube(self):
        return []

    def get_robot_intructions(self, solve_instruction_list):
        return []

    def blink_led(self, times, duration=0.5):
        for _ in range(times):
            GPIO.output(self.LIGHT_PIN, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.LIGHT_PIN, GPIO.LOW)
            time.sleep(duration)

    def doStuffs(self):
        # Get the button pressed
        if self.button.pressed():
            print("Button pressed!")
            time.sleep(1)


if __name__ == '__main__':
    # GPIO pins
    BUTTON_PIN = 17
    LIGHT_PIN = 27

    # Setup GPIO pins
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Run
    try:
        print("Running cubert...")
        robot = RubiksCubeRobot(BUTTON_PIN, LIGHT_PIN)
        while True:
            robot.doStuffs()
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
