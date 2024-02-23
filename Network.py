import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size + 2 * padding - kernel_size) // stride) + 1


class ResidualBlock(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout):
        super(ResidualBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ])
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.fc(x)


class ResidualBlock_Network(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, hidden_size, dropout):
        super(ResidualBlock_Network, self).__init__(input_obs_space, features_dim)

        # Dynamically calculate the flattened size of the observation space
        flattened_obs_space = int(np.prod(input_obs_space.shape))

        # Input layer
        self.network = nn.Sequential(
            nn.Linear(flattened_obs_space, 54),
            nn.BatchNorm1d(54),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(54, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            # Dynamically create the specified number of ResidualBlocks
            ResidualBlock(2, hidden_size, dropout),

            # Output layer
            nn.Linear(hidden_size, features_dim)
        )

    def forward(self, observations):
        return self.network(torch.flatten(observations, start_dim=1))


class ResidualBlock_512_20_Dropout(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         hidden_size=100,
                         dropout=0.2)


class ResidualBlock_1024_50_Dropout(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         hidden_size=1024,
                         dropout=0.5)

