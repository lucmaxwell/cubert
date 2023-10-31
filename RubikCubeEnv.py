import gym
from gym import spaces
import torch
import numpy as np

from RubikCube import RubikCube, Face

# Global cube size
CUBE_SIZE = 3

# Define the action space dictionary globally
TOTAL_FACES = 6
TOTAL_SPINS = 4
action_space_dict = {
    'face': spaces.Discrete(TOTAL_FACES),  # 6 faces
    'spin': spaces.Discrete(TOTAL_SPINS),  # 0 for left, 1 for right, 2 for down, 3 for up
    'row_or_col': spaces.Discrete(CUBE_SIZE)  # Row or column index
}


class RubiksCubeEnv(gym.Env):
    def __init__(self):
        super(RubiksCubeEnv, self).__init__()

        self.cube = RubikCube(CUBE_SIZE)

        # Define action and observation space
        self.action_space = spaces.Dict(action_space_dict)
        self.observation_space = spaces.Box(low=0, high=5, shape=(6, CUBE_SIZE, CUBE_SIZE), dtype=np.int8)

        self.reward_range = (-1, 1)
        self.reset()

    def reset(self, **kwargs):
        self.cube = RubikCube(CUBE_SIZE)
        return self._get_observation()

    def step(self, action):
        # Extract the action
        face = Face(action['face'])
        direction = action['spin']
        row_or_col = action['row_or_col']

        # Apply the action
        if direction == 0:
            self.cube.rotate_row_left(face, row_or_col)
        elif direction == 1:
            self.cube.rotate_row_right(face, row_or_col)
        elif direction == 2:
            self.cube.rotate_column_down(face, row_or_col)
        elif direction == 3:
            self.cube.rotate_column_up(face, row_or_col)

        # Calculate reward based on the number of correct squares
        correct_squares = self.cube.count_correct_squares()
        reward = correct_squares / (self.cube.size * self.cube.size * 6)  # Normalize

        # Check if the cube is solved
        done = self.cube.is_solved()
        if done:
            reward += 1.0  # Extra reward for solving the cube

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        observation = np.zeros((6, 3, 3), dtype=np.int8)
        for i, face in enumerate(Face):
            observation[i] = self.cube.get_face_colors(face)
        return observation
