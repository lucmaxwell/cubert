import gymnasium
import numpy as np
from stable_baselines3.common.env_checker import check_env

from RubikCube import RubikCube, Face, SquareColour

# Global cube size
CUBE_SIZE = 3

# Define the action space dictionary globally
TOTAL_FACES = 6
TOTAL_SPINS = 4 # Left, right, down, up
TOTAL_ACTIONS = TOTAL_FACES * TOTAL_SPINS * CUBE_SIZE # Each face has the cube size selection of row or column

NUM_SCRAMBLE = 1

TOTAL_SQUARES = CUBE_SIZE**2 * TOTAL_FACES


def decode_action(action):
    face = action // (TOTAL_SPINS * CUBE_SIZE)
    remaining = action % (TOTAL_SPINS * CUBE_SIZE)
    spin = remaining // CUBE_SIZE
    row_or_col = remaining % CUBE_SIZE
    return face, spin, row_or_col


class RubiksCubeEnv(gymnasium.Env):
    def __init__(self):
        super(RubiksCubeEnv, self).__init__()

        self.cube = RubikCube(CUBE_SIZE)

        # Define action and observation space
        self.action_space = gymnasium.spaces.Discrete(TOTAL_ACTIONS)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=(TOTAL_FACES - 1),
            shape=(TOTAL_FACES, CUBE_SIZE, CUBE_SIZE),
            dtype=np.uint8)

        # Define the reward range
        self.reward_range = (-1, 1)


    def reset(self, **kwargs):
        self.cube.reset()
        self.cube.scramble(NUM_SCRAMBLE)
        return self._get_observation(), {}

    def step(self, action):
        # Decode the action
        face, spin, row_or_col = decode_action(action)

        # print("Initial state:")
        # self.cube.print_cube_state()
        # print(f"Action face {face} spin {spin} row/col {row_or_col}")

        # Apply the action
        if spin == 0:
            self.cube.rotate_row_left(face, row_or_col)
        elif spin == 1:
            self.cube.rotate_row_right(face, row_or_col)
        elif spin == 2:
            self.cube.rotate_column_down(face, row_or_col)
        elif spin == 3:
            self.cube.rotate_column_up(face, row_or_col)

        # Calculate reward based on the number of correct squares
        done = self.cube.is_solved()
        reward = 1 if done else -1

        # print("Action result:")
        # self.cube.print_cube_state()

        # Return
        return self._get_observation(), reward, done, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            self.cube.print_cube_state()

    def _get_observation(self):
        observation = np.zeros((TOTAL_FACES, CUBE_SIZE, CUBE_SIZE), dtype=np.uint8)
        for i, face in enumerate(Face):
            observation[i] = self.cube.get_face_colors(face)
        return observation


if __name__ == '__main__':
    # Create an instance of the environment
    env = RubiksCubeEnv()
    check_env(env)

    # Action space sample
    print(env.action_space.sample())
