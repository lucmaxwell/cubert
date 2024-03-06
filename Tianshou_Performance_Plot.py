import os

import torch
from matplotlib import pyplot as plt
from tianshou.policy import DQNPolicy
from torch.optim import Adam

from Tianshou_Model_Validation import episode
from tianshou.utils.net.common import Net
from Network import Tianshou_Network
from RubikCubeEnv import RubiksCubeEnv


def performance_plot(policy, env, model_name):
    # Test the trained agent
    solved_count_list = []
    num_scrambles = range(1, 13 + 1)
    for num_scramble in num_scrambles:

        # Focus on the first 3 scramble only
        num_episodes = 1000
        if num_scramble > 4:
            num_episodes = 100

        # Solve the puzzles
        solved_count = 0
        for _ in range(num_episodes):
            solved = episode(policy, env, num_scramble)
            if solved:
                solved_count += 1

        # Solved percentage
        solved_percentage = (solved_count / num_episodes) * 100

        print(f"Scramble {num_scramble:<2d}: {solved_percentage}% solved")

        solved_count_list.append(solved_percentage)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(num_scrambles, solved_count_list, marker='o')
    plt.title(model_name)
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
    MODEL_NAME = "DQN_Tianshou"
    #MODEL_NAME = "DQN_Tianshou_Vector"
    save_path = os.path.join('Training', 'Saved Models')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup environment
    env = RubiksCubeEnv()
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    net = Tianshou_Network(state_shape, action_shape).to(device)
    optim = Adam(net.parameters(), lr=1e-3)
    policy = DQNPolicy(net, optim)

    # Load the saved policy state
    model_file_path = os.path.join(save_path, MODEL_NAME + '.pth')
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        policy.load_state_dict(torch.load(model_file_path))
        net.eval()  # Set to eval mode

        performance_plot(policy, env, MODEL_NAME)

    else:
        print(f"File does not exist {model_file_path}")
