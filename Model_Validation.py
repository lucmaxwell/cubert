import concurrent.futures

import pandas as pd
from matplotlib import pyplot as plt

from RubikCubeEnv import RubiksCubeEnv
from UtilityFunctions import load_model_PPO, load_model_DQN


def episode(model, num_scramble, multiple_attempts=False):
    # Create the environment
    env = RubiksCubeEnv(num_scramble=num_scramble)

    # Solve the puzzle
    # Allow multiple attempts
    env.scramble(num_scramble)
    done = False
    count = 0
    while count < 3 and not done:
        count += 1

        # Solve the cube
        obs, _ = env.reset()
        while not done:
            # Determine action and take step
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)

        # Check if the cube has been solved
        done = env.is_solved()

        if not multiple_attempts:
            break

    # Return
    return done


def evaluate_scramble(model, num_scramble, num_episodes=100, multiple_attempts=False):
    # Solve the puzzles
    solved_percentage = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit episodes as futures
        futures = [executor.submit(episode, model, num_scramble, multiple_attempts=multiple_attempts) for _ in range(num_episodes)]

        # Wait for all futures to complete and get results
        results = [future.result() for future in futures]

        solved_count = sum(results)
        solved_percentage = (solved_count / num_episodes) * 100

    # Return
    return solved_percentage


def evaluate_model(model):
    evaluation_results = []
    num_scrambles = range(1, 13 + 1)
    for num_scramble in num_scrambles:
        # Solve the puzzles
        solved_percentage = evaluate_scramble(model, num_scramble)
        evaluation_results.append(solved_percentage)
        print(f"Scramble {num_scramble:<2d}: {solved_percentage}% solved")

    # Return
    return num_scrambles, evaluation_results


def test(model, plot_title, num_episodes=1000):
    # Test the trained agent
    solved_count_list = []
    num_scrambles = range(1, 13 + 1)
    for num_scramble in num_scrambles:

        # Focus on the first 3 scramble only
        actual_num_episodes = num_episodes
        if num_scramble > 4:
            actual_num_episodes = 100

        # Solve the puzzles
        solved_percentage = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit episodes as futures
            futures = [executor.submit(episode, model, num_scramble, _) for _ in range(actual_num_episodes)]

            # Wait for all futures to complete and get results
            results = [future.result() for future in futures]

            solved_count = sum(results)
            solved_percentage = (solved_count / actual_num_episodes) * 100

        print(f"Scramble {num_scramble:<2d}: {solved_percentage}% solved")

        solved_count_list.append(solved_percentage)

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
    MODEL_NAME = "dqn_training_gen_10"

    # Create a new model by default
    test_model, environment, _ = load_model_DQN(MODEL_NAME, num_scramble=1)

    test(test_model, environment, MODEL_NAME)
