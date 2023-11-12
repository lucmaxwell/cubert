import os
import random
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch

from RubikCubeEnv import RubiksCubeEnv, NUM_SCRAMBLE


# Hyperparameters
GAMMA = 0.99 # Discount rate
EPSILON_START = 0
EPSILON_DECAY = 0
BATCH_SIZE = 32

NUM_EPISODES = 100
NUM_STEP_PER_EPISODE = 1000
NUM_PARALLEL_ENV = 10
VERBOSE = 1
MODEL_NAME = "ppo_rubik_model_gen_1"


if __name__ == '__main__':
    start_time = time.time()

    print(torch.__version__)

    save_path = os.path.join('Training', 'Saved Models')
    log_path = os.path.join('Training', 'Logs')

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()
    wrapper_env = DummyVecEnv([lambda: Monitor(RubiksCubeEnv())])
    # wrapper_env = SubprocVecEnv([lambda: Monitor(RubiksCubeEnv()) for _ in range(NUM_PARALLEL_ENV)])

    # Create a new model by default
    #training_model = PPO('MlpPolicy', wrapper_env, verbose=VERBOSE, tensorboard_log=log_path, device="cuda")
    #training_model = PPO('MlpPolicy', env, verbose=VERBOSE, device="cuda")
    training_model = PPO('MlpPolicy', wrapper_env, learning_rate=0.000003, verbose=VERBOSE, tensorboard_log=log_path, device="cuda")

    # Check if a saved model exists and load it
    model_file_path = os.path.join(save_path, MODEL_NAME)
    if os.path.isfile(model_file_path + ".zip"):
        print("Loading existing model...")
        training_model = PPO.load(model_file_path, env=env)

    # CUDA available
    print(f"CUDA is available {torch.cuda.is_available()}")

    # Print whether training is using CUDA
    device = training_model.policy.device
    print(f"Training on device: {device}")

    # Train and save the agent
    training_model.learn(total_timesteps=NUM_EPISODES * NUM_STEP_PER_EPISODE)
    for episode in range(NUM_EPISODES):
        env.scramble()

        # Try moves
        epsilon = EPSILON_START
        for _ in range(NUM_STEP_PER_EPISODE):
            obs, _ = env.reset()

            # Epsilon-greedy action selection
            action = env.action_space.sample()
            # Stable path
            if np.random.rand() > epsilon:
                action, _ = training_model.predict(obs)

            obs, reward, done, _, _ = env.step(action)

            # Update epsilon
            epsilon *= EPSILON_DECAY

    # Save the model after each level of complexity
    training_model.save(model_file_path)
    print(f"Model saved.")

    # Release resource
    env.close()

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Evaluate the model
    evaluate_policy(training_model, env)
