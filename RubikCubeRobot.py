from RubikCube import RubikCube, Face

from enum import Enum


class RobotAction(Enum):
    SPIN_BASE_CLOCKWISE = 0
    SPIN_BASE_COUNTERCLOCKWISE = 1
    GRIP_AND_FLIP = 2
    SPIN_CUBE_CLOCKWISE = 3
    SPIN_CUBE_COUNTERCLOCKWISE = 4


def reorient_to_face(target_face):
    # The robot grib location indicate the top

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
    def __init__(self):
        # Initialize the cube's state
        self.cube = RubikCube(3)
        self.instructions = []

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


if __name__ == '__main__':
    robot = RubiksCubeRobot()
    robot.update_cube_state("new_state_from_input_or_model")
    robot.solve_cube()
    instructions = robot.get_instructions()
    print("Instructions to solve the cube:", instructions)
    robot.execute_instructions()
