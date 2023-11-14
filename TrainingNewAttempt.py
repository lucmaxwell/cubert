import os
import random
import time
from stable_baselines3 import PPO
import torch

from RubikCubeEnv import RubiksCubeEnv
from workSpace.Testing import testing, validate_reinforcement

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
    TOTAL_EPISODE = 100 * 5
    for episode in range(TOTAL_EPISODE):
        # Random number of scramble
        num_scramble_list = range(1, 2 + 1)
        num_scramble = random.choice(num_scramble_list)
        print(f"Episode: {episode + 1}, number of scramble: {num_scramble}")

        # Setup to run the episode
        MAX_STEPS_PER_EPISODE = env.get_max_steps_per_episode()
        env.set_num_scramble(num_scramble)

        # New scramble cube
        env.scramble()

        # Solve the Rubik's Cube
        num_steps_to_solved = 0
        total_steps = 0
        attempt = 0
        solved = False
        while not solved:
            obs, _ = env.reset()

            for step in range(MAX_STEPS_PER_EPISODE):
                total_steps += 1

                # Action and reward
                action, _ = training_model.predict(obs)
                obs, reward, solved, _, _ = env.step(action)

                if solved:
                    num_steps_to_solved = step + 1
                    break

            # Learn
            if solved:
                # There is a better solution
                if num_steps_to_solved > num_scramble:
                    print(
                        f"Not optimal solution. Number of steps solved: {num_steps_to_solved}")
                    solved = False

                # Found and learned the most efficient solution
                # Validate the reinforcement
                elif not validate_reinforcement(env, training_model):
                    print(f"Validate reinforcement failed.")
                    solved = False

        training_model.learn(total_timesteps=total_steps)

        # Print episode information
        print(f"Number of steps solved: {num_steps_to_solved}, Total steps: {total_steps}")

        # Checkpoint save
        if (episode + 1) % 100 == 0:
            checkpoint_model_name = MODEL_NAME + "_" + str(episode + 1)
            checkpoint_path = os.path.join(save_path, checkpoint_model_name + ".zip")
            training_model.save(checkpoint_path)

            testing(checkpoint_model_name, checkpoint_model_name)

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
    testing(MODEL_NAME, MODEL_NAME)
