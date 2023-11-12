import math
import os
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
    num_scramble_list = range(3, 3 + 1)
    for num_scramble in num_scramble_list:
        print(f"Number of scrambles = {num_scramble}")
        env.set_num_scramble(num_scramble)

        TOTAL_EPISODE = 100
        MAX_STEPS_PER_EPISODE = int(math.ceil(num_scramble * 2.5))
        MAX_STEPS = TOTAL_EPISODE * MAX_STEPS_PER_EPISODE

        for episode in range(TOTAL_EPISODE):
            episode_reward = 0
            num_steps_to_solved = 0

            # New scramble cube
            env.scramble()

            solved = False
            while not solved:
                obs, _ = env.reset()

                for step in range(MAX_STEPS_PER_EPISODE):
                    # Action and reward
                    action, _ = training_model.predict(obs)
                    obs, reward, solved, _, _ = env.step(action)

                    episode_reward += reward

                    if solved:
                        num_steps_to_solved = step + 1
                        break

            training_model.learn(total_timesteps=MAX_STEPS_PER_EPISODE)

            # Print episode information
            print(f"Episode: {episode + 1}, Number of steps solved: {num_steps_to_solved + 1}, Total Reward: {episode_reward}")

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
