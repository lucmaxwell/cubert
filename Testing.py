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

MODEL_NAME = "best_child"


if __name__ == '__main__':
    start_time = time.time()

    save_path = os.path.join('Training', 'Saved Models')

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()
    wrapper_env = DummyVecEnv([lambda: Monitor(env)])

    # Load the model
    print("Loading existing model...")
    model_file_path = os.path.join(save_path, MODEL_NAME)
    training_model = PPO.load(model_file_path, env=wrapper_env)

    # Test the trained agent
    NUM_RUN = 100
    solved_count = 0
    for iteration in range(NUM_RUN):
        env.scramble()
        obs, _ = env.reset()

        done = False
        move_count = 0
        while not done and move_count < NUM_SCRAMBLE:
            action, _states = training_model.predict(obs)
            obs, rewards, done, _, _ = env.step(action)

            # Update move count
            move_count += 1

        if done:
            solved_count += 1

    print(f"Solved {solved_count / NUM_RUN * 100}%")

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
