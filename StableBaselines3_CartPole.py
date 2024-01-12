import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn


class AscendingNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim, hidden_size=256):
        super(AscendingNetwork, self).__init__(observation_space, features_dim)

        flattened_obs_space = int(np.prod(observation_space.shape))

        self.network = nn.Sequential(
            nn.Linear(flattened_obs_space, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(),

            nn.Linear(hidden_size * 2, hidden_size * 3),
            nn.LeakyReLU(),

            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LeakyReLU(),

            nn.Linear(hidden_size * 2, 256),
            nn.LeakyReLU(),

            nn.Linear(256, features_dim)
        )

    def forward(self, observations):

        # Unpack the tuple here if obs is a tuple with more than one element
        if isinstance(observations, tuple):
            observations, _ = observations  # This will ignore the second part (empty dictionary in your case)

        # Flatten the observations
        obs_flat = torch.flatten(observations, start_dim=1)
        return self.network(obs_flat)


def make_env(env_id):
    def _init():
        #env = gym.make(env_id, render_mode="human")
        env = gym.make(env_id)
        return env
    return _init


if __name__ == "__main__":
    start_time = time.time()

    # Initialize your environment
    env_id = "CartPole-v1"
    num_envs = 10  # Number of parallel environments
    envs = SubprocVecEnv([make_env(env_id) for i in range(num_envs)])

    # Define policy_kwargs to include your custom network
    policy_kwargs = dict(
        features_extractor_class=AscendingNetwork,
        features_extractor_kwargs=dict(features_dim=envs.action_space.n)
    )

    # Initialize the DQN model with your custom network
    model = DQN("MlpPolicy", envs, policy_kwargs=policy_kwargs, verbose=2)

    # Train the model
    model.learn(total_timesteps=1_000_000)

    # End time and elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    # Test the trained agent
    env = gym.make("CartPole-v1", render_mode="human")
    print("Validating...")

    obs = env.reset()
    obs = obs[0]
    for _ in range(10_000):

        env.render()

        action, _states = model.predict(obs)

        obs, reward, done, info, _ = env.step(action)

        if done:
            obs = env.reset()
            obs = obs[0]
