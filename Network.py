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


class ResidualBlock_64_50_Dropout(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         hidden_size=64,
                         dropout=0.5)


class ResidualBlock_1024_50_Dropout(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         hidden_size=1024,
                         dropout=0.5)


class The_best_thing_that_we_have(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, num_layers, hidden_size, dropout):
        super(The_best_thing_that_we_have, self).__init__(input_obs_space, features_dim)
        # Define the convolutional network layers
        TOTAL_FACES = 6
        last_conv_layer_dim = 128
        self.conv_layers = nn.Sequential(
            # Convolutional Layer 1: Input is the 6 faces of the cube
            nn.Conv2d(in_channels=TOTAL_FACES, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Convolutional Layer 2
            nn.Conv2d(in_channels=64, out_channels=last_conv_layer_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(last_conv_layer_dim),
            nn.ReLU(),
        )
        # Define the network layers
        cube_size = input_obs_space.shape[1]
        conv_output_size = calculate_conv_output_size(cube_size, 3, 1, 1)
        conv_output_size = calculate_conv_output_size(conv_output_size, 3, 1, 1)
        flattened_conv = conv_output_size * conv_output_size * last_conv_layer_dim
        self.network = nn.Sequential(
            nn.Linear(flattened_conv, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            ResidualBlock(num_layers, hidden_size, dropout),
            # Output layer
            nn.Linear(hidden_size, features_dim)
        )
    def forward(self, observations):
        # Convolution layer input
        conv_out = self.conv_layers(observations)
        # Deep layers
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.network(conv_out)


class Machine_1024_50_Dropout(The_best_thing_that_we_have):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=2,
                         hidden_size=1024,
                         dropout=0.5)

class Machine_1024_20_Dropout(The_best_thing_that_we_have):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=2,
                         hidden_size=1024,
                         dropout=0.2)

class Machine_2048_50_Dropout(The_best_thing_that_we_have):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=2,
                         hidden_size=2048,
                         dropout=0.5)

class Machine_2048_20_Dropout(The_best_thing_that_we_have):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=2,
                         hidden_size=2048,
                         dropout=0.2)








# class Tianshou_Network(nn.Module):
#     def __init__(self, state_shape, action_shape):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
#             nn.Linear(128, 128), nn.ReLU(inplace=True),
#             nn.Linear(128, 128), nn.ReLU(inplace=True),
#             nn.Linear(128, np.prod(action_shape)),
#         )
#
#     def forward(self, obs, state=None, info={}):
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs, dtype=torch.float)
#         batch = obs.shape[0]
#         logits = self.model(obs.view(batch, -1))
#         return logits, state


class Tianshou_Network_Old(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        return self.model(obs.flatten(1)), state


class Tianshou_Residual_Block(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout):
        super(Tianshou_Residual_Block, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ])
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.fc(x)


class Tianshou_Network(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        flattened_obs_space = int(np.prod(state_shape))

        hidden_size = 1024
        dropout = 0.2

        # Common features
        self.feature = nn.Sequential(
            # Input layers
            nn.Linear(np.prod(flattened_obs_space), hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            # Features
            Tianshou_Residual_Block(4, hidden_size, dropout),
        )

        # Outputs a single value (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Outputs advantage for each action (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, np.prod(action_shape))
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=next(self.parameters()).device)

        obs_flat = torch.flatten(obs, start_dim=1)
        features = self.feature(obs_flat)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values, state
