import torch
import torch.nn as nn


class RubikLearningAgent(nn.Module):
    def __init__(self):
        super(RubikLearningAgent, self).__init__()

        # First Convolutional layer: Captures 3x3 patterns on each face
        self.conv1 = nn.Conv2d(6, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization

        # First Fully Connected Layer: 128 units to capture complex features
        self.fc1 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Batch Normalization

        # Output Layer: 12 possible actions (rotations)
        self.fc2 = nn.Linear(128, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = torch.flatten(x, 1)  # Flatten while keeping the batch dimension

        x = self.fc1(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.fc2(x)
        return x

    def get_expected_future_reward(self, next_state, immediate_reward, discount_factor=0.99):
        # Ensure the code works on the same device as the model
        device = next(self.parameters()).device

        # Convert to tensor and add a batch dimension
        next_state_tensor = torch.tensor(next_state, device=device).float().unsqueeze(0)

        # Get the Q-values for the next state
        with torch.no_grad():
            next_action_values = self.forward(next_state_tensor)

        # Calculate expected future reward
        # y = r + γ * max over a' of Q(s', a')
        max_next_action_value = next_action_values.max().item()
        expected_reward = immediate_reward + discount_factor * max_next_action_value

        # Return
        return expected_reward
