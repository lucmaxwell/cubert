import os

import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch.nn.functional as F

from CustomCallback import SaveOnEqualRewardCallback
from RubikCubeEnv import RubiksCubeEnv


def load_model_PPO(model_name, num_scramble=1):
    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')
    model_log_path = os.path.join('Training', 'Logs\\' + model_name)

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv(num_scramble=num_scramble)

    # Create a new model by default
    training_model = None
    model_file_path = os.path.join(save_path, model_name + ".zip")
    if os.path.isfile(model_file_path):
        print(f"Loading existing model from {model_file_path}")
        training_model = PPO.load(model_file_path, env=env, verbose=2, tensorboard_log=model_log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = PPO('MlpPolicy', env=env, verbose=2, tensorboard_log=model_log_path, device="cuda")

    # Callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=os.path.join(save_path, model_name),
                                             name_prefix=model_name)
    eval_env = Monitor(RubiksCubeEnv(num_scramble=num_scramble))
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(save_path, model_name + "_best_model"),
                                 log_path=os.path.join(model_log_path, model_name + "_best_model"), eval_freq=10000,
                                 deterministic=True)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Return
    return training_model, env, callback


class Network(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(Network, self).__init__(observation_space, features_dim)

        # Dynamically calculate the flattened size of the observation space
        flattened_obs_space = int(np.prod(observation_space.shape))

        # Define the network layers
        self.layer1 = nn.Linear(flattened_obs_space, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, features_dim)

    def forward(self, observations):
        # Flatten the observations
        obs_flat = observations.flatten(start_dim=1)

        x = F.relu(self.layer1(obs_flat))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def load_model_DQN(model_name, num_scramble=1):
    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')
    model_log_path = os.path.join('Training', 'Logs\\' + model_name)

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()

    # Define the policy kwargs with custom feature extractor
    policy_kwargs = dict(features_extractor_class=Network)

    # Create a new model by default
    training_model = None
    model_file_path = os.path.join(save_path, model_name + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = DQN.load(model_file_path, env=env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log=model_log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = DQN('MlpPolicy', env=env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log=model_log_path, device="cuda")

    # Callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=os.path.join(save_path, model_name),
                                             name_prefix=model_name)
    eval_env = Monitor(RubiksCubeEnv(num_scramble=num_scramble))
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(save_path, model_name + "_best_model"),
                                 log_path=os.path.join(model_log_path, model_name + "_best_model"), eval_freq=10000,
                                 deterministic=True)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Return
    return training_model, env, callback


def save_model(model_name, model):
    save_path = os.path.join('Training', 'Saved Models')

    model_file_path = os.path.join(save_path, model_name + ".zip")

    model.save(model_file_path)
    print(f"Model {model_name} saved. Path: {model_file_path}")


if __name__ == '__main__':
    env = RubiksCubeEnv()

    network = Network(env.observation_space)
    print(network)

