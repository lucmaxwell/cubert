import os

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from torch.optim import Adam
from torch import load

from Network import Tianshou_Network
from RubikCubeEnv import RubikCubeEnv, decode_action


def episode(policy, env, num_scramble):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Solve
    done = False
    count = 0
    obs = env.scramble(num_scramble)

    # Convert observation to tensor and move to the specified device
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

    while not done and count < 100:
        count += 1

        #batch = Batch(obs=np.array([obs]), info={})

        batch = Batch(obs=obs_tensor, info={})
        action = policy(batch).act[0]
        obs, _, done, _, _ = env.step(action)

        # Update obs_tensor for the next iteration
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # Update with new observation

    done = env.is_solved()

    return done


def episode_with_render(policy, env, num_scramble):
    # Solve
    done = False
    count = 0
    obs = env.scramble(num_scramble)
    env.render()
    while not done and count < 100:
        count += 1

        batch = Batch(obs=np.array([obs]), info={})
        action = policy(batch).act[0]
        obs, _, done, _, _ = env.step(action)

        print(f"Count {count}")
        print(f"Decode action {decode_action(action)}")
        env.render()

    if not env.is_solved():
        done = False

    return done


def run_episodes(policy, env, num_scramble):
    solved = True
    solved_count = 0
    while solved and solved_count < 100:
        solved = episode(policy, env, num_scramble)

        if solved:
            solved_count += 1

    return solved_count


if __name__ == '__main__':
    #MODEL_NAME = "DQN_Tianshou.pth"
    MODEL_NAME = "DQN_Tianshou_Vector.pth"

    # Setup environment
    num_scramble = 1
    env = RubikCubeEnv(num_scramble=num_scramble)

    # Setup model and policy as during training
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Tianshou_Network(state_shape, action_shape)
    optim = Adam(net.parameters(), lr=1e-3)
    policy = DQNPolicy(net, optim, estimation_step=10)  # Optimizer and loss function are not needed for inference

    # Load the saved policy state
    model_path = 'Training/Saved Models/' + MODEL_NAME
    policy.load_state_dict(load(model_path))
    net.eval() # Set to eval mode

    # Solve
    print(f"num_scramble={num_scramble}")
    while True:
        print(f"Done: {episode_with_render(policy, env, num_scramble)}")
        input("Press Enter to continue...")
