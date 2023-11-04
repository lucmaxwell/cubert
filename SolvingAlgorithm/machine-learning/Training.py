import os
import random
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch

from RubikCubeEnv import RubiksCubeEnv, NUM_SCRAMBLE


# Hyperparameters
GAMMA = 0.99 # Discount rate
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
NUM_EPISODES = 100
BATCH_SIZE = 32

NUM_PARALLEL_ENV = 20
VERBOSE = 0
MODEL_NAME = "ppo_rubik_model_gen_1"


if __name__ == '__main__':
    start_time = time.time()

    print(torch.__version__)

    save_path = os.path.join('Training', 'Saved Models')
    #log_path = os.path.join('Training', 'Logs')

    # Create the environment and vector for parallel environments
    #env = DummyVecEnv([lambda: Monitor(RubiksCubeEnv())])
    env = SubprocVecEnv([lambda: Monitor(RubiksCubeEnv()) for _ in range(NUM_PARALLEL_ENV)])

    # Create a new model by default
    #training_model = PPO('MlpPolicy', env, verbose=VERBOSE, tensorboard_log=log_path)
    training_model = PPO('MlpPolicy', env, verbose=VERBOSE, device="cuda")

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
    for k in range(1, 2):  # 13 levels of complexity
        for episode in range(1000):  # 100 episodes per level
            # Set the number of scramble moves
            NUM_SCRAMBLE = random.randint(1, k)

            # Reset environment (and scramble the cube)
            obs = env.reset()

            for step in range(k):  # Maximum k steps to solve the cube
                action, _ = training_model.predict(obs)
                obs, rewards, done, info = env.step(action)

                # Learn from the single step
                training_model.learn(total_timesteps=1)

            print(f"Episode {episode + 1} completed.")

        # Save the model after each level of complexity
        training_model.save(model_file_path)
        print(f"Model saved after {k} levels of complexity.")

    # Release resource
    env.close()

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Evaluate the model
    evaluate_policy(training_model, RubiksCubeEnv())
