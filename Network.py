import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn



def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size + 2 * padding - kernel_size) // stride) + 1


class ResidualBlock(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(ResidualBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            ])
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.fc(x)  # Add input to the output (residual connection)


class ResidualBlock_Network(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, num_layers, hidden_size):
        super(ResidualBlock_Network, self).__init__(input_obs_space, features_dim)

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
            nn.Dropout(0.1),

            ResidualBlock(num_layers, hidden_size),

            # Output layer
            nn.Linear(hidden_size, features_dim)
        )

    def forward(self, observations):
        # Convolution layer input
        conv_out = self.conv_layers(observations)

        # Deep layers
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.network(conv_out)


class ResidualBlock_2Layers_2048(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space, features_dim, num_layers=2, hidden_size=2048)


class ResidualBlock_2Layers_4096(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space, features_dim, num_layers=2, hidden_size=4096)


class ResidualBlock_3Layers_2048(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space, features_dim, num_layers=3, hidden_size=2048)















