import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from RubikCubeEnv import RubiksCubeEnv, NUM_SCRAMBLE
from RubikLearningAgent import RubikLearningAgent

# Test parameter
NUM_ITERATION = 10000

MODEL_NAME = "ppo_rubik_model_gen_1"


if __name__ == '__main__':

    save_path = os.path.join('Training', 'Saved Models')

    # Create the environment and vector for parallel environments
    env = DummyVecEnv([lambda: Monitor(RubiksCubeEnv())])

    # Load the model
    print("Loading existing model...")
    model_file_path = os.path.join(save_path, MODEL_NAME)
    training_model = PPO.load(model_file_path, env=env)

    # Test the trained agent
    obs = env.reset()
    NUM_RUN = 1000
    solved_count = 0
    for iteration in range(NUM_RUN):
        done = False
        move_count = 0
        total_reward_collected = 0
        while not done and move_count < NUM_SCRAMBLE * 2:
            action, _states = training_model.predict(obs)
            obs, rewards, done, info = env.step(action)

            # Update reward
            total_reward_collected += rewards

            # Update move count
            move_count += 1

        if done:
            solved_count += 1

    print(f"Solved {solved_count / NUM_RUN * 100}%")
