import numpy as np
import torch
import torch.optim as optim
from gym import spaces

from RubikCubeEnv import RubiksCubeEnv, CUBE_SIZE
from RubikLearningAgent import RubikLearningAgent

# Parameters
GAMMA = 0.99 # Discount factor
LEARNING_RATE = 0.001
EPSILON_START = 1.0 # Exploration rate
EPSILON_DECAY = 0.995
NUM_EPISODES = 1000

if __name__ == '__main__':
    env = RubiksCubeEnv()
    model = RubikLearningAgent()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Train
    gamma = GAMMA
    epsilon = EPSILON_START
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0  # Keep track of total reward per episode

        while not done:
            # Epsilon-greedy action selection
            action = {
                'face': np.random.randint(6),  # 6 faces
                'spin': np.random.randint(4),  # 4 spin directions
                'row_or_col': np.random.randint(CUBE_SIZE)  # CUBE_SIZE rows/cols
            }
            # Pursue a more stable path
            if np.random.rand() > epsilon:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)

                # Map action_idx to an action in your action space
                action = {
                    'face': action_idx // (4 * CUBE_SIZE),
                    'spin': (action_idx % (4 * CUBE_SIZE)) // CUBE_SIZE,
                    'row_or_col': action_idx % CUBE_SIZE
                }

            next_state, reward, done, _ = env.step(action)

            # Get the expected future reward
            expected_future_reward = model.get_expected_future_reward(next_state, reward, gamma)

            # Calculate loss
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.LongTensor([action])

            predicted = model(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze()
            loss = criterion(predicted, torch.FloatTensor([expected_future_reward]))

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        # Decay epsilon
        if epsilon > 0.1:
            epsilon *= EPSILON_DECAY

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

