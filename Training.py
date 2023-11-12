import math
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch

from RubikCubeEnv import RubiksCubeEnv

# Hyperparameters
NUM_PARALLEL_ENV = 10
VERBOSE = 1

TOTAL_TIME_STEPS = 1000000
NUM_EPISODES = 1000

NUM_SCRAMBLE = 1
MAX_STEPS_ATTEMPT = int(math.ceil(NUM_SCRAMBLE * 2.5))

MODEL_NAME = "ppo_1_scramble"

if __name__ == '__main__':
    start_time = time.time()

    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')
    log_path = os.path.join('Training', 'Logs\\' + MODEL_NAME)

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()

    # Create a new model by default
    model_file_path = os.path.join(save_path, MODEL_NAME)
    if os.path.isfile(model_file_path + ".zip"):
        print("Loading existing model...")
        training_model = PPO.load(model_file_path, env=env, verbose=VERBOSE, tensorboard_log=log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = PPO('MlpPolicy', env=env, verbose=VERBOSE, tensorboard_log=log_path, device="cuda")

    # Training
    total_steps = 0
    training_model.learn(total_timesteps=TOTAL_TIME_STEPS)
    for episode in range(NUM_EPISODES):
        env.scramble()

        solved = False
        num_steps = 0
        while not solved:
            # Reset the Rubik's Cube
            obs, _ = env.reset()

            while num_steps < MAX_STEPS_ATTEMPT:
                # Action and reward
                action, _ = training_model.predict(obs)
                obs, reward, done, _, _ = env.step(action)

                # Update the number of steps taken
                num_steps += 1

        # Done training
        total_steps += num_steps
        if total_steps >= TOTAL_TIME_STEPS:
            break

    # Save the model
    training_model.save(model_file_path)
    print(f"Model saved. Path: {model_file_path}")

    # Release resource
    env.close()

    # End time and elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Evaluate the model
    eval_env = RubiksCubeEnv()
    mean_reward, std_reward = evaluate_policy(training_model, eval_env, n_eval_episodes=10)
    print(f"Evaluation: Mean reward: {mean_reward}, Std: {std_reward}")
