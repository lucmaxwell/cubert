import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

from RubikCubeEnv import RubiksCubeEnv
from RubikLearningAgent import RubikLearningAgent


# Hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
NUM_EPISODES = 1000
BATCH_SIZE = 32


if __name__ == '__main__':
    start_time = time.time()

    # Initialize environment and agent
    env = RubiksCubeEnv()
    agent = RubikLearningAgent()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Initialize variables for tracking performance
    states = []
    actions = []
    rewards = []
    next_states = []

    # Load pre-trained model if available
    model_path = "rubik_model.pth"
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))

    # Training loop
    epsilon = EPSILON_START
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                sampled_action = env.action_space.sample()
                action = {
                    'face': sampled_action['face'],
                    'spin': sampled_action['spin'],
                    'row_or_col': sampled_action['row_or_col']}
            else:
                q_values = agent(torch.tensor(state, dtype=torch.float32))  # Modified line
                action_key = q_values.argmax().item()
                action = {
                    'face': action_key // 12,
                    'spin': (action_key % 12) // 3,
                    'row_or_col': action_key % 3}

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Append to lists
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

            # If batch is large enough, perform an update
            if len(states) >= BATCH_SIZE:
                state_batch = torch.tensor(states, dtype=torch.float32)
                next_state_batch = torch.tensor(next_states, dtype=torch.float32)
                # Convert actions and rewards to tensors as needed

                # Calculate Q-values and loss using batches
                q_values = agent(state_batch)
                # ... (Your existing code for calculating target and loss)

                # Clear the lists
                states.clear()
                actions.clear()
                rewards.clear()
                next_states.clear()

                # Perform the optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Update Q-values using Bellman equation
            q_values = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            target = reward + GAMMA * agent(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)).max().item()
            prediction = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[action]

            loss = criterion(prediction, target)
            episode_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        # Update epsilon
        epsilon *= EPSILON_DECAY

        # Logging
        print(f"Episode: {episode}, Total Reward: {total_reward}, Loss: {episode_loss}, Epsilon: {epsilon}")
        rewards.append(total_reward)
        losses.append(episode_loss)

        # Save model checkpoint after every 100 episodes
        if episode % 100 == 0:
            torch.save(agent.state_dict(), f"rubik_model_checkpoint_{episode}.pth")

    # Save final model
    torch.save(agent.state_dict(), "rubik_model_final.pth")

    # Generate and save performance reports
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    np.save(f"rewards_{timestamp}.npy", np.array(rewards))
    np.save(f"losses_{timestamp}.npy", np.array(losses))

    # Generate and save plots
    plt.figure()
    plt.plot(rewards)
    plt.title("Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"rewards_plot_{timestamp}.png")

    plt.figure()
    plt.plot(losses)
    plt.title("Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(f"losses_plot_{timestamp}.png")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
