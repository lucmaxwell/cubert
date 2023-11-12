import math
import os
import time

from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from RubikCubeEnv import RubiksCubeEnv

MODEL_NAME = "ppo_episode_train"


def testing(model_name):
    start_time = time.time()

    save_path = os.path.join('Training', 'Saved Models')

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()

    # Load the model
    model_file_path = os.path.join(save_path, model_name + ".zip")
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    print("Loading existing model...")
    training_model = PPO.load(model_file_path, env=env)

    # Test the trained agent
    solved_count_list = []
    num_scrambles = range(1, 13 + 1)
    for num_scramble in num_scrambles:
        MAX_STEPS = int(math.ceil(num_scramble * 2.5))
        env.set_num_scramble(num_scramble)

        solved_count = 0
        for _ in range(100):
            env.scramble()
            obs, _ = env.reset()

            # Solve
            solved = False
            num_steps = 0
            while not solved and num_steps < MAX_STEPS:
                # Action and reward
                action, _ = training_model.predict(obs, deterministic=True)
                obs, reward, solved, _, _ = env.step(action)

                num_steps += 1

            if solved:
                solved_count += 1

        solved_count_list.append(solved_count)

        print(f"Scramble {num_scramble:<2d} {solved_count}%")

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(num_scrambles, solved_count_list, marker='o')
    plt.title('Solved Counts vs Number of Scrambles')
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
    testing(MODEL_NAME)
