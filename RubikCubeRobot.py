import time

import GPIO

from RubikCube import RubikCube, Face
from Button import Button

from enum import Enum


class RobotAction(Enum):
    SPIN_BASE_CLOCKWISE = 0
    SPIN_BASE_COUNTERCLOCKWISE = 1
    GRIP_AND_FLIP = 2
    SPIN_CUBE_CLOCKWISE = 3
    SPIN_CUBE_COUNTERCLOCKWISE = 4


def get_moves_to_face(target_face):
    # The robot grib location indicate the bottom

    # The front face will be the bottom where the camera is looking at the cube
    # Already the front face, no action
    instructions = []
    if target_face == Face.Back:
        instructions.append(RobotAction.GRIP_AND_FLIP)
        instructions.append(RobotAction.GRIP_AND_FLIP)
    elif target_face == Face.Right:
        _rotate_cube_y_clockwise()
    elif target_face == Face.Left:
        _rotate_cube_y_counter_clockwise()
    elif target_face == Face.Top:
        _rotate_cube_x_clockwise()
    elif target_face == Face.Bottom:
        _rotate_cube_x_counter_clockwise()

    # Return
    return instructions


class RubiksCubeRobot:
    BUTTON_PIN = 17

    def __init__(self):
        # Initialize the button
        self.button = Button(self.BUTTON_PIN, GPIO.HIGH)

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

    def doStuffs(self):
        # Get the button pressed
        if self.button.pressed():
            hold_time = self.button.hold_time()
            pressed_count = 0

            # Count the number of pressed or hold time
            BUFFER_TIME = 0.05
            reading_start_time = time.time()
            reading_time = time.time()
            while reading_time - reading_start_time < BUFFER_TIME:
                # Button released
                if not self.button.pressed():
                    # Start the countdown
                    reading_start_time = time.time()

                    # Register a press and release
                    pressed_count += 1

                # Button pressed
                else:
                    # Start the countdown
                    reading_start_time = time.time()

                    # Register the pressed time
                    hold_time = self.button.hold_time()

                # Update the reading time
                reading_time = time.time()

            # Solve
            if pressed_count == 1:
                pass
            # Remember scrambled
            elif pressed_count == 2:
                pass
            # Solve to remember scrambled
            elif pressed_count == 3:
                pass
            # Turn off base light
            elif pressed_count == 1 and 1 <= hold_time < 2:
                pass
            # Play victory song
            elif hold_time >= 2:
                pass


if __name__ == '__main__':
    # Setup GPIO pins

    # Run
    robot = RubiksCubeRobot()
    while True:
        robot.doStuffs()
