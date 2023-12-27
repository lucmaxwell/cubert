import os
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch import nn

from Network import AscendingNetwork
from RubikCubeEnv import RubiksCubeEnv
from Model_Validation import evaluate_model


# class AscendingNetwork(BaseFeaturesExtractor):
#     def __init__(self, input_obs_space, features_dim, hidden_size=512):
#         super(AscendingNetwork, self).__init__(input_obs_space, features_dim)
#
#         # Dynamically calculate the flattened size of the observation space
#         flattened_obs_space = int(np.prod(input_obs_space.shape))
#
#         # Define the network layers
#         self.network = nn.Sequential(
#             nn.Linear(flattened_obs_space, hidden_size),
#             nn.LeakyReLU(),
#
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(hidden_size * 2),
#
#             nn.Linear(hidden_size * 2, hidden_size * 3),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(hidden_size * 3),
#
#             nn.Linear(hidden_size * 3, hidden_size * 2),
#             nn.LeakyReLU(),
#
#             nn.Linear(hidden_size * 2, 256),
#             nn.LeakyReLU(),
#
#             nn.Linear(256, features_dim)
#         )
#
#     def forward(self, observations):
#         # Flatten the observations
#         obs_flat = torch.flatten(observations, start_dim=1)
#
#         return self.network(obs_flat)


def make_env(num_scrambles):
    def _init():
        env = RubiksCubeEnv(num_scramble=num_scrambles)
        return env
    return _init


NUM_ENVS = 5
TOTAL_STEPS = 100_000
MODEL_NAME = "dqn_ascending"

NUM_SCRAMBLES = 1

if __name__ == '__main__':
    start_time = time.time()

    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')
    model_log_path = os.path.join('Training', 'Logs\\' + MODEL_NAME)

    # Create the environment and vector for parallel environments
    envs = SubprocVecEnv([make_env(env_id) for i in range(num_envs)])

    # Define the policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=AscendingNetwork,
        features_extractor_kwargs=dict(features_dim=env.action_space.n)
    )

    # Create a new model or load model if already existed
    training_model = None
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = DQN.load(model_file_path, env=envs, verbose=2,
                                  tensorboard_log=model_log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = DQN('MlpPolicy', env=envs, policy_kwargs=policy_kwargs, verbose=2,
                             tensorboard_log=model_log_path, device="cuda")

    # Training
    training_model.learn(total_timesteps=TOTAL_STEPS)

    # Save the model
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    training_model.save(model_file_path)
    print(f"Model {MODEL_NAME} saved. Path: {model_file_path}")

    # End time and elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Evaluate
    start_time = time.time()
    evaluate_model(training_model, MODEL_NAME)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Release resource
    envs.close()
