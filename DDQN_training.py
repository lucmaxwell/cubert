import os

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv

from Network import ResidualBlock_2Layers_2048, ResidualBlock_3Layers_2048, \
    ResidualBlock_4Layers_1024, LSTM_4Layers_1024, ResidualBlock_2Layers_4096, ResidualBlock_3Layers_4096, DoubleDQN
from RubikCubeEnv import RubiksCubeEnv
from Training_Utility_Functions import train_and_evaluate

network_configuration = ResidualBlock_2Layers_4096

NUM_SCRAMBLES = 3

NUM_STEPS = 100_000


def make_env(num_scrambles):
    def _init():
        env = RubiksCubeEnv(num_scramble=num_scrambles)
        return env
    return _init


if __name__ == '__main__':
    MODEL_NAME = "DDQN_" + network_configuration.__name__ + "_SubprocVecEnv"

    save_path = os.path.join('Training', 'Saved Models')

    # Ensure the directory exists
    plot_folder_name = 'training_plot'
    os.makedirs(plot_folder_name, exist_ok=True)

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv(num_scramble=NUM_SCRAMBLES)

    # Define the policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=network_configuration,
        features_extractor_kwargs=dict(features_dim=env.action_space.n)
    )

    # Create a new model or load model if already existed
    training_model = None
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = DoubleDQN.load(model_file_path,
                                  env=env,
                                  verbose=0,
                                  device="cuda")
    else:
        print("Creating a new model...")
        training_model = DoubleDQN(policy='MlpPolicy',
                             env=env,
                             policy_kwargs=policy_kwargs,
                             verbose=0,
                             device="cuda")

    # Learn and evaluate
    train_and_evaluate(
        training_model,
        save_path,
        MODEL_NAME,
        NUM_STEPS,
        NUM_SCRAMBLES,
        plot_folder_name
    )

    # Release resource
    env.close()
