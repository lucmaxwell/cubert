import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import torch.nn.functional

from RubikCubeEnv import RubiksCubeEnv, NUM_SCRAMBLE
from RubikLearningAgent import RubikLearningAgent


# Hyperparameters
GAMMA = 0.99 # Discount rate
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
NUM_EPISODES = 100
BATCH_SIZE = 32


if __name__ == '__main__':
    start_time = time.time()

    # Initialize environment and agent
    env = RubiksCubeEnv()
    training_model = RubikLearningAgent()
    optimizer = optim.Adam(training_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Initialize variables for tracking performance
    states = []
    actions = []
    rewards = []
    next_states = []

    # Load pre-trained model if available
    model_path = "rubik_model.pth"
    if os.path.exists(model_path):
        training_model.load_state_dict(torch.load(model_path))

    # Training loop
    epsilon = EPSILON_START
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0

        # Print the initial state
        print(f"Initial state episode {episode}:")
        env.render()

        move_count = 0
        while not done and move_count < NUM_SCRAMBLE:
            action = env.action_space.sample()
            n_state, reward, done,  = env.step(action)
            total_reward += reward
        print('Episode:{} Score:{}'.format(episode, score))


            # Epsilon-greedy action selection
            sampled_action = env.action_space.sample()
            action = {
                'face': sampled_action['face'],
                'spin': sampled_action['spin'],
                'row_or_col': sampled_action['row_or_col']}
            # Stable path
            if np.random.rand() > epsilon:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = training_model(state_tensor)
                _, action = torch.max(q_values, dim=1)
                action = action.item()

            # Take action and get reward
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Q-function update
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            action_tensor = torch.LongTensor([action]).unsqueeze(0)

            q_values = training_model(state_tensor)
            current_q_value = q_values.gather(1, action_tensor).squeeze()

            with torch.no_grad():
                next_q_values = training_model(next_state_tensor)
                max_next_q_value = next_q_values.max(1)[0]
                target_q_value = reward + (GAMMA * max_next_q_value) * (not done)

            loss = torch.nn.functional.mse_loss(current_q_value, target_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update
            move_count += 1
            state = next_state

        # Update epsilon
        epsilon *= EPSILON_DECAY

        # Logging
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
        rewards.append(total_reward)

        # Save model checkpoint after every 100 episodes
        if episode % 100 == 0:
            torch.save(training_model.state_dict(), f"rubik_model_checkpoint_{episode}.pth")

    # Release resource
    env.close()

    # Save final model
    torch.save(training_model.state_dict(), "rubik_model_final.pth")

    # Generate and save performance reports
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    np.save(f"rewards_{timestamp}.npy", np.array(rewards))

    # Generate and save plots
    plt.figure()
    plt.plot(rewards)
    plt.title("Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"rewards_plot_{timestamp}.png")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
