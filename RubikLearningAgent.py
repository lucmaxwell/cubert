import torch
import torch.nn as nn
import torch.nn.functional


class RubikLearningAgent(nn.Module):
    def __init__(self):
        super(RubikLearningAgent, self).__init__()

        # Input layer: 54 squares * 6 possible colors = 324
        # Hidden layers: two layers of 128 neurons each
        # Output layer: 72 possible actions (6 faces * 3 rows/cols * 2 directions)
        self.fc1 = nn.Linear(324, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 72)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
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
        # MSELoss
        max_next_action_value = next_action_values.max().item()
        expected_reward = immediate_reward + discount_factor * max_next_action_value

        # Return
        return expected_reward
