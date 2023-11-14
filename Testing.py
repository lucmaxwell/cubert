import math
import os
import time

from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from RubikCubeEnv import RubiksCubeEnv
from UtilityFunctions import load_model_PPO, load_model_DQN


def validate_reinforcement(model, env, num_episodes=10):
    success_count = 0
    optimal_success_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        # New scramble cube
        env.scramble()

        obs, _ = env.reset()

        solved = False
        steps = 0
        while not solved and steps < env.get_max_steps_per_episode():
            steps += 1

            action, _ = model.predict(obs, deterministic=True)
            obs, _, solved, _ = env.step(action)

        if solved:
            success_count += 1

            # Optimally solved
            if steps <= env.get_num_scramble():
                optimal_success_count += 1

        total_steps += steps

    average_steps = total_steps / num_episodes
    success_rate = success_count / num_episodes
    optimal_success_rate = optimal_success_count / num_episodes

    # Return
    return average_steps, success_rate, optimal_success_rate


def test(model, env, plot_title):
    # Test the trained agent
    solved_count_list = []
    num_scrambles = range(1, 13 + 1)
    for num_scramble in num_scrambles:
        env.set_num_scramble(num_scramble)
        MAX_STEPS = env.get_max_steps()

        solved_count = 0
        for _ in range(100):
            obs = env.scramble()

            # Solve
            solved = False
            num_steps = 0
            while not solved and num_steps < MAX_STEPS:
                # Action and reward
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, solved, _, _ = env.step(action)

                num_steps += 1

            if solved:
                solved_count += 1

        solved_count_list.append(solved_count)

        print(f"Scramble {num_scramble:<2d} {solved_count}%")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(num_scrambles, solved_count_list, marker='o')
    plt.title(plot_title)
    plt.xlabel('Number of Scrambles')
    plt.ylabel('Solved Counts')
    plt.xticks(num_scrambles)
    plt.grid(True)

    # Annotate each point with its value
    for i, count in enumerate(solved_count_list):
        plt.annotate(str(count), (num_scrambles[i], solved_count_list[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    MODEL_NAME = "dqn_train_100"

    save_path = os.path.join('Training', 'Saved Models')

    # Create the environment and vector for parallel environments
    environment = RubiksCubeEnv()

    # Create a new model by default
    test_model = load_model_DQN(save_path, MODEL_NAME, environment)

    test(test_model, environment, MODEL_NAME)
