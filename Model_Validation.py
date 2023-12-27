import concurrent.futures

import pandas as pd
from matplotlib import pyplot as plt

from RubikCubeEnv import RubiksCubeEnv
from UtilityFunctions import load_model_PPO, load_model_DQN


def episode(model, num_scramble, id):
    # Create the environment
    env = RubiksCubeEnv(num_scramble=num_scramble)

    # Solve the puzzle
    done = False
    obs = env.scramble(num_scramble)
    while not done:
        # Determine action and take step
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    # Check if the cube has been solved
    done = env.is_solved()

    # Return
    return done


def evaluate_model(model, model_name, num_episodes=1000):
    evaluation_results = []

    for num_scramble in range(1, 13 + 1):

        # Focus on the first 3 scramble only
        actual_num_episodes = num_episodes
        if num_scramble > 3:
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
            evaluation_results.append({
                'num_scramble': num_scramble,
                'solved_percentage': solved_percentage
            })

        print(f"Scramble {num_scramble:<2d}: {solved_percentage}% solved")

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(evaluation_results)
    df.to_csv("./Plot/" + model_name + ".csv", index=False)
    print(f"Evaluation results saved.")


def test(model, env, plot_title):
    # Test the trained agent
    solved_count_list = []
    num_scrambles = range(1, 13 + 1)
    for num_scramble in num_scrambles:
        env.set_num_scramble(num_scramble)
        MAX_STEPS = env.get_max_steps()

        solved_count = 0
        for _ in range(100):
            obs = env.scramble(num_scramble)

            #print("Original:")
            #env.render()

            # Solve
            done = False
            while not done and env.get_current_num_steps() < MAX_STEPS:
                # Action and reward
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)

            if env.is_solved():
                solved_count += 1

            #print("Result:")
            #env.render()
            #print(f"Number of steps to solve: {env.get_current_num_steps()}")

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
    MODEL_NAME = "dqn_training_gen_10"

    # Create a new model by default
    test_model, environment, _ = load_model_DQN(MODEL_NAME, num_scramble=1)

    test(test_model, environment, MODEL_NAME)
