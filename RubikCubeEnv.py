import math
import random

import gymnasium
import numpy as np
from stable_baselines3.common.env_checker import check_env

from RubikCube import RubikCube, Face, SquareColour

# Define the action space dictionary globally
TOTAL_FACES = 6
TOTAL_SPINS = 2  # Clockwise, Counter-clockwise


def decode_action(action):
    face = action // TOTAL_SPINS
    spin = action % TOTAL_SPINS
    return face, spin


class RubiksCubeEnv(gymnasium.Env):
    def __init__(self, num_scramble=1, cube_size=3, max_steps_per_episode=100):
        super(RubiksCubeEnv, self).__init__()

        # Define action and observation space
        self.action_space = gymnasium.spaces.Discrete(TOTAL_FACES * TOTAL_SPINS)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=(TOTAL_FACES - 1),
            shape=(TOTAL_FACES, cube_size, cube_size),
            dtype=np.uint8)

        # Define the reward range
        self.reward_range = (-1, 1)

        self.cube_size = cube_size
        self.num_scramble = num_scramble
        self.current_num_scramble = num_scramble

        self.max_steps_per_episode = max_steps_per_episode
        self.current_num_steps = 0
        self.episode_reward = 0

        # Choosing number of scramble weight
        self.weights = [n ** 2 for n in range(1, self.num_scramble + 1)]

        # Create and scramble the Rubik's Cube
        self.cube = RubikCube(self.cube_size)
        self.scramble(self.current_num_scramble)

    def set_num_scramble(self, num_scramble):
        self.num_scramble = num_scramble
        self.reward_range = (-1, self.num_scramble)
        self.weights = [n**2 for n in range(1, self.num_scramble + 1)]

    def get_num_scramble(self):
        return self.num_scramble

    def get_max_steps(self):
        #return int(math.ceil(self.num_scramble * 2.5))
        return self.num_scramble * 10

    def is_solved(self):
        return self.cube.is_solved()

    def get_current_num_steps(self):
        return self.current_num_steps

    def scramble(self, num_scramble):
        self.current_num_scramble = num_scramble

        self.cube = RubikCube(self.cube_size)
        self.cube.scramble(self.current_num_scramble)

        self.current_num_steps = 0
        self.episode_reward = 0

        return self._get_observation()

    def reset(self, **kwargs):
        self.current_num_scramble = random.choices(range(1, self.num_scramble + 1), weights=self.weights, k=1)[0]

        #self.current_num_scramble = self.num_scramble

        self.cube = RubikCube(self.cube_size)
        self.cube.scramble(self.current_num_scramble)

        self.current_num_steps = 0
        self.episode_reward = 0

        return self._get_observation(), {}

    def step(self, action):
        self.current_num_steps += 1

        # Decode the action
        face, spin = decode_action(action)
        face = Face(face)

        # Apply the action
        if spin == 0:
            self.cube.rotate_clockwise(face)
        elif spin == 1:
            self.cube.rotate_counter_clockwise(face)

        # print(f"Action face: {face} spin: {spin}")
        # self.render()

        # Calculate reward based on the number of correct squares
        done = self.cube.is_solved()
        reward = 1 if done and self.current_num_steps <= self.num_scramble else -1

        # Update the episode reward
        self.episode_reward += reward

        # New scramble
        if self.current_num_steps >= self.get_max_steps():
            done = True

        # Return
        return self._get_observation(), reward, done, False, {}

    def render(self):
        self.cube.print_cube_state()
        # pass

    def _get_observation(self):
        observation = np.zeros((TOTAL_FACES, self.cube_size, self.cube_size), dtype=np.uint8)
        for i, face in enumerate(Face):
            observation[i, :, :] = self.cube.get_face_colors(face)

        # Ensure that the observation matches the defined observation space
        assert self.observation_space.contains(observation), "The generated observation is out of bounds!"

        return observation


if __name__ == '__main__':
    # Create an instance of the environment
    env = RubiksCubeEnv()

    # Action space sample
    print(env.action_space.sample())

    # Initial state and reset
    print("Initial state:")
    env.render()

    # Reset state
    env.reset()
    print("Reset state should be the same as initial:")
    env.render()

    check_env(env)
