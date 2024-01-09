import numpy as np
import torch
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
        return x + self.fc(x)  # Add input to the output (residual connection)


class ResidualBlock_Network(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, num_layers, hidden_size, dropout):
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


class LSTM_Network(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, num_layers, hidden_size, dropout):
        super(LSTM_Network, self).__init__(input_obs_space, features_dim)

        # Define the convolutional network layers
        intermediate_layer_dim = 128
        last_conv_layer_dim = 256
        self.conv_layers = nn.Sequential(
            # Convolutional Layer 1: Input is the 6 faces of the cube
            nn.Conv2d(in_channels=6, out_channels=intermediate_layer_dim, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_layer_dim),
            nn.ReLU(),

            # Convolutional Layer 2
            nn.Conv2d(in_channels=intermediate_layer_dim, out_channels=last_conv_layer_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(last_conv_layer_dim),
            nn.ReLU(),
        )

        # Define the network layers
        cube_size = input_obs_space.shape[1]
        conv_output_size = calculate_conv_output_size(cube_size, 2, 1, 1)
        conv_output_size = calculate_conv_output_size(conv_output_size, 3, 1, 1)
        flattened_conv = conv_output_size * conv_output_size * last_conv_layer_dim
        self.lstm = nn.LSTM(input_size=flattened_conv, hidden_size=hidden_size, num_layers=2, dropout=dropout, batch_first=True)

        self.residual_block = ResidualBlock(num_layers, hidden_size, dropout)

        self.output_fc = nn.Linear(hidden_size, features_dim)

    def forward(self, observations):
        # Convolution layer input
        conv_out = self.conv_layers(observations)

        # Flatten the convolutional output
        batch_size = conv_out.size(0)
        conv_out_flat = conv_out.view(batch_size, -1)

        # Reshape for LSTM - treat the entire set of features from convolutional layers as one sequence
        lstm_in = conv_out_flat.unsqueeze(1)

        # LSTM layer input
        lstm_out, _ = self.lstm(lstm_in)
        # Take the output for the last time step
        lstm_out = lstm_out[:, -1, :]

        # Apply residual block on LSTM output
        res_out = self.residual_block(lstm_out)

        # Fully connected layer input
        output = self.output_fc(res_out)
        return output


class ResidualBlock_2Layers_4096(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space, features_dim, num_layers=2, hidden_size=4096, dropout=0.2)


class ResidualBlock_4Layers_1024(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=4,
                         hidden_size=1024,
                         dropout=0.7)


class LSTM_4Layers_1024(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=4,
                         hidden_size=1024,
                         dropout=0.7)














class ResidualBlock_2Layers_2048(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=2,
                         hidden_size=2048,
                         dropout=0.7)

class ResidualBlock_3Layers_2048(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=3,
                         hidden_size=2048,
                         dropout=0.7)


class ResidualBlock_2Layers_4096(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=2,
                         hidden_size=4096,
                         dropout=0.8)


class ResidualBlock_3Layers_4096(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=3,
                         hidden_size=4096,
                         dropout=0.8)







