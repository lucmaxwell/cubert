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
                         dropout=0.5)


class ResidualBlock_3Layers_4096(ResidualBlock_Network):
    def __init__(self, input_obs_space, features_dim):
        super().__init__(input_obs_space,
                         features_dim,
                         num_layers=3,
                         hidden_size=4096,
                         dropout=0.5)





class DoubleDQN(DQN):
    def __init__(self, *args, **kwargs):
        # Initialize the learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy.optimizer, gamma=0.99)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Ensure that replay buffer exists
        assert self.replay_buffer is not None, "No replay buffer was created for training"

        # Switch to train mode (this affects batch norm / dropout)
        # Update learning rate according to schedule
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Follow DDQN update rule
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.policy.q_net_target(replay_data.next_observations)

                # Compute q-values for the next observation using the online q net
                next_q_values_online = self.policy.q_net(replay_data.next_observations)

                # Select action with online network
                next_actions_online = next_q_values_online.argmax(dim=1)

                # Estimate the q-values for the selected actions using target q network
                next_q_values = torch.gather(next_q_values, dim=1, index=next_actions_online.unsqueeze(-1)).squeeze(-1)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values.unsqueeze(1)

            # Get current Q-values estimates
            current_q_values = self.policy.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            assert current_q_values.shape == target_q_values.shape, "Shape mismatch between current and target Q-values"

            # Compute loss (L2 or Huber loss)
            loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)

            # Log the loss value
            losses.append(loss.item())

            # Optimize the q-network
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update learning rate according to schedule
            self.lr_scheduler.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))




'''
def __init__(self, *args, **kwargs):
        super(DoubleDQN, self).__init__(*args, **kwargs)
        # Initialize your optimizer here
        self.policy.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        # Initialize the learning rate scheduler here
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.policy.optimizer, step_size=100, gamma=0.9)

 # Update the learning rate scheduler
            self.lr_scheduler.step()
'''

'''

            
            
            

        for gradient_step in range(gradient_steps):
            # Sample from replay buffer
            replay_data = replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Follow DDQN update rule
            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.policy.q_net_target(replay_data.next_observations)

                # use current model to select the action with maximal q value
                max_actions = torch.argmax(self.q_net(replay_data.next_observations), dim=1)

                # evaluate q value of that action using fixed target network
                next_q_values = torch.gather(next_q_values, dim=1, index=max_actions.unsqueeze(-1))

                # Compute the target Q values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                print(f"Target q value: {target_q_values.shape}")

                '''
'''
                 # Compute the next Q-values using the target network
                next_q_values = self.policy.q_net_target(replay_data.next_observations)
                
                # Select action according to the policy network
                next_actions = self.policy.q_net(replay_data.next_observations).argmax(dim=1)
                next_q_values = torch.gather(next_q_values, dim=1, index=next_actions.unsqueeze(-1)).squeeze(-1)
                
                # Compute the target Q values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                '''

'''

            # Optimize the policy network
            self.policy.optimizer.zero_grad()
            q_values = self.policy.q_net(replay_data.observations)

            # Retrieve the Q-values for the actions from the replay buffer
            action_q_values = torch.gather(q_values, dim=1, index=replay_data.actions).squeeze(-1)

            # Calculate loss
            loss = torch.nn.functional.mse_loss(action_q_values, target_q_values)
            loss.backward()
            self.policy.optimizer.step()

            self._on_step()

'''

