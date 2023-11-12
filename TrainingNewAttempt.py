import math
import os
import random
import time
from stable_baselines3 import PPO
import torch

from RubikCubeEnv import RubiksCubeEnv
from Testing import testing

# Hyperparameters
VERBOSE = 0

MODEL_NAME = "ppo_episode_train"

if __name__ == '__main__':
    start_time = time.time()

    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()

    # Create a new model by default
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = PPO.load(model_file_path, env=env, verbose=VERBOSE, device="cuda")
    else:
        print("Creating a new model...")
        training_model = PPO('MlpPolicy', env=env, verbose=VERBOSE, device="cuda")

    # Training
    TOTAL_EPISODE = 100
    for episode in range(TOTAL_EPISODE):
        # Random number of scramble
        num_scramble_list = range(1, 2 + 1)
        num_scramble = random.choice(num_scramble_list)
        print(f"Episode: {episode + 1}, number of scramble: {num_scramble}")

        # Setup to run the episode
        MAX_STEPS_PER_EPISODE = int(math.ceil(num_scramble * 2.5))
        env.set_num_scramble(num_scramble)
        episode_reward = 0
        num_steps_to_solved = 0
        training_model.ent_coef = 0.0

        # New scramble cube
        env.scramble()

        # Solve the Rubik's Cube
        attempt = 0
        solved = False
        while not solved:
            attempt += 1

            obs, _ = env.reset()

            for step in range(MAX_STEPS_PER_EPISODE):
                # Action and reward
                action, _ = training_model.predict(obs)
                obs, reward, solved, _, _ = env.step(action)

                episode_reward += reward

                if solved:
                    num_steps_to_solved = step + 1
                    break

            # If not solved, increase the exploration
            if not solved and attempt % 10 == 0:
                if training_model.ent_coef < 1.0:
                    training_model.ent_coef += 0.01

            # Learn
            if solved:
                training_model.learn(total_timesteps=num_steps_to_solved)

                # There is a better solution
                if num_steps_to_solved > num_scramble:
                    print(f"Not optimal solution. Number of steps solved: {num_steps_to_solved}, ent_coef: {training_model.ent_coef}")
                    solved = False

        # Print episode information
        print(f"Number of steps solved: {num_steps_to_solved}, Total Reward: {episode_reward}")

    # Save the model
    training_model.save(model_file_path)
    print(f"Model saved. Path: {model_file_path}")

    # Release resource
    env.close()

    # End time and elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Test
    testing(MODEL_NAME)
