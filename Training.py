import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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

    save_path = os.path.join('Training', 'Saved Models')
    log_path = os.path.join('Training', 'Logs')

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()
    env = DummyVecEnv([lambda: env])

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

    # Train the agent
    model.learn(total_timesteps=10000)

    # Test the trained agent
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()




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
        env.reset()
        done = False
        total_reward = 0

        # Print the initial state
        print(f"Initial state episode {episode}:")
        env.render()

        move_count = 0
        while not done and move_count < NUM_SCRAMBLE:
            action = env.action_space.sample()
            n_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Action taken
            move_count += 1
        print('Episode:{} Score:{}'.format(episode, total_reward))


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
