import math
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from stable_baselines3.common.monitor import Monitor

from RubikCubeEnv import RubiksCubeEnv
from Testing import testing

# Hyperparameters
NUM_PARALLEL_ENV = 10
VERBOSE = 1

TOTAL_TIME_STEPS = 100000

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
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = PPO.load(model_file_path, env=env, verbose=VERBOSE, tensorboard_log=log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = PPO('MlpPolicy', env=env, verbose=VERBOSE, tensorboard_log=log_path, device="cuda")

    # Training
    total_steps = 0
    training_model.learn(total_timesteps=TOTAL_TIME_STEPS)
    while total_steps >= TOTAL_TIME_STEPS:
        env.scramble()

        solved = False
        num_steps = 0
        while not solved:
            # Reset the Rubik's Cube to the scrambled state
            obs, _ = env.reset()

            while num_steps < MAX_STEPS_ATTEMPT:
                # Action and reward
                action, _ = training_model.predict(obs)
                obs, reward, _, _, _ = env.step(action)

                solved = env.cube.is_solved()

                # Update the number of steps taken
                num_steps += 1

        # Update the total number of steps
        total_steps += num_steps

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

    # Evaluate the model
    num_scrambles = range(1, 13 + 1)
    for num_scramble in num_scrambles:
        eval_env = RubiksCubeEnv(num_scramble=num_scramble)
        monitored_eval_env = Monitor(eval_env)

        mean_reward, std_reward = evaluate_policy(training_model, monitored_eval_env, n_eval_episodes=100)
        print(f"Evaluation {num_scramble}: Mean reward: {mean_reward}, Std: {std_reward}")
