import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


env = gym.make('CartPole-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
memory_size = 10000


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


memory = ReplayBuffer(memory_size)

model = DQN(state_dim, 128, action_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

num_episodes = 100
for episode in range(num_episodes):

    state = env.reset()
    state = state[0]

    for step in range(100):

        env.render()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = model(torch.from_numpy(state).float().unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(action_dim)]], dtype=torch.long)

        next_state, reward, done, _, _ = env.step(action.item())

        # Store in replay memory
        memory.push(state, action, reward, next_state, done)

        state = next_state

        # Start training after enough samples are in the memory
        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            # Extract states, actions, etc. from the batch
            # and perform a gradient descent step

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, min_epsilon)







