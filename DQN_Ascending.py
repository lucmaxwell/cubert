import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from RubikCubeEnv import RubiksCubeEnv
from Model_Validation import evaluate_model

TOTAL_STEPS = 100_000
MODEL_NAME = "dqn_LSTM_1024"

NUM_SCRAMBLES = 2


def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return ((input_size + 2 * padding - kernel_size) // stride) + 1


class Network(BaseFeaturesExtractor):
    def __init__(self, input_obs_space, features_dim, hidden_size=1024):
        super(Network, self).__init__(input_obs_space, features_dim)

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
        self.lstm = nn.LSTM(input_size=flattened_conv, hidden_size=hidden_size, batch_first=True)

        self.residual_block = ResidualBlock(hidden_size)

        self.fc = nn.Linear(hidden_size, features_dim)

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
        output = self.fc(res_out)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.fc(x)  # Add input to the output (residual connection)


if __name__ == '__main__':
    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')
    model_log_path = os.path.join('Training', 'Logs\\' + MODEL_NAME)

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv(num_scramble=NUM_SCRAMBLES)

    # Define the policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=Network,
        features_extractor_kwargs=dict(features_dim=env.action_space.n)
    )

    # Create a new model or load model if already existed
    training_model = None
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = DQN.load(model_file_path, env=env, verbose=2,
                                  tensorboard_log=model_log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = DQN('MlpPolicy', env=env, policy_kwargs=policy_kwargs, verbose=2,
                             tensorboard_log=model_log_path, device="cuda")

    # Keep learning
    run_count = 0
    solved_percentage = 0.0
    while solved_percentage < 100.0:
        run_count += 1
        start_time = time.time()

        # Training
        training_model.learn(total_timesteps=TOTAL_STEPS)

        # Save the model
        model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
        training_model.save(model_file_path)
        print(f"Model {MODEL_NAME} saved. Path: {model_file_path}")

        # Elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        # Evaluate
        start_time = time.time()
        num_scrambles, evaluation_results = evaluate_model(training_model, NUM_SCRAMBLES)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(num_scrambles, evaluation_results, marker='o')
        plt.title(f"{MODEL_NAME} scramble={NUM_SCRAMBLES} run_count={run_count}")
        plt.xlabel('Number of Scrambles')
        plt.ylabel('Solved Counts')
        plt.xticks(num_scrambles)
        plt.grid(True)

        # Annotate each point with its value
        for i, count in enumerate(evaluation_results):
            plt.annotate(str(count), (num_scrambles[i], evaluation_results[i]), textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        # Show the plot
        plt.show()

        # Update the interested solved result
        solved_percentage = evaluation_results[NUM_SCRAMBLES - 1]

    # Release resource
    env.close()
