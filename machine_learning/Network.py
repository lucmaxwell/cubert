import numpy as np
import torch
from torch import nn


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
        action_space = np.prod(action_shape)

        hidden_size = 1024
        dropout = 0.2

        # Common features
        self.feature = nn.Sequential(
            # Input layers
            nn.Linear(np.prod(flattened_obs_space), hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            # Features
            Tianshou_Residual_Block(3, hidden_size, dropout),
        )

        # Outputs a single value (V)
        self.value_stream = nn.Sequential(
            Tianshou_Residual_Block(2, hidden_size, dropout),
            nn.Linear(hidden_size, 1)
        )

        # Outputs advantage for each action (A)
        self.advantage_stream = nn.Sequential(
            Tianshou_Residual_Block(4, hidden_size, dropout),
            nn.Linear(hidden_size, action_space)
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
