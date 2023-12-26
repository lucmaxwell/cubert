import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class StandardNetwork(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, hidden_size=256):
        super(StandardNetwork, self).__init__(input_obs_space, features_dim)

        # Dynamically calculate the flattened size of the observation space
        flattened_obs_space = int(np.prod(input_obs_space.shape))

        # Define the network layers
        self.network = nn.Sequential(
            nn.Linear(flattened_obs_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size*2),
            nn.Linear(hidden_size*2, hidden_size*2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, features_dim)
        )

    def forward(self, observations):
        # Flatten the observations
        obs_flat = torch.flatten(observations, start_dim=1)

        return self.network(obs_flat)


class AscendingNetwork(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, hidden_size=256):
        super(AscendingNetwork, self).__init__(input_obs_space, features_dim)

        # Dynamically calculate the flattened size of the observation space
        flattened_obs_space = int(np.prod(input_obs_space.shape))

        # Define the network layers
        self.network = nn.Sequential(
            nn.Linear(flattened_obs_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size * 3),
            nn.ReLU(),
            nn.Linear(hidden_size * 3, features_dim)
        )

    def forward(self, observations):
        # Flatten the observations
        obs_flat = torch.flatten(observations, start_dim=1)

        return self.network(obs_flat)


class DescendingNetwork(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, hidden_size=256):
        super(DescendingNetwork, self).__init__(input_obs_space, features_dim)

        # Dynamically calculate the flattened size of the observation space
        flattened_obs_space = int(np.prod(input_obs_space.shape))

        # Define the network layers
        self.network = nn.Sequential(
            nn.Linear(flattened_obs_space, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size*2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, features_dim)
        )

    def forward(self, observations):
        # Flatten the observations
        obs_flat = torch.flatten(observations, start_dim=1)

        return self.network(obs_flat)
