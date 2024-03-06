import math
import os

import torch
import numpy as np
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from torch import nn
from torch.optim import Adam, AdamW

from Network import Tianshou_Network
from RubikCubeEnv import RubikCubeEnv
from Tianshou_Model_Validation import run_episodes


NUM_SCRAMBLES = 5
NUM_ENVS = 4


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    MODEL_NAME = "DQN_Tianshou_Vector"
    #MODEL_NAME = "DQN_Tianshou_Vector_1024_4"

    save_path = os.path.join('Training', 'Saved Models')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up the environment
    env = RubikCubeEnv(num_scramble=NUM_SCRAMBLES)

    # Set up the network and policy
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Tianshou_Network(state_shape, action_shape).to(device)

    # Parameters
    optim = AdamW(net.parameters(), lr=2e-4)
    policy = DQNPolicy(
        net,
        optim,
        estimation_step=int(math.ceil(NUM_SCRAMBLES * 2.5)),
        is_double=True
    )

    # Set up the collector and replay buffer
    buffer = VectorReplayBuffer(10000, NUM_ENVS)
    train_envs = SubprocVectorEnv(
        [lambda: RubikCubeEnv(num_scramble=NUM_SCRAMBLES) for _ in range(NUM_ENVS)]
    )
    train_collector = Collector(policy, train_envs, buffer)
    test_envs = SubprocVectorEnv(
        [lambda: RubikCubeEnv(num_scramble=NUM_SCRAMBLES) for _ in range(4)]
    )
    test_collector = Collector(policy, test_envs)

    # Load progress
    model_file_path = os.path.join(save_path, MODEL_NAME + '.pth')
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        policy.load_state_dict(torch.load(model_file_path))

    # Train the model
    print(f"Training model: {MODEL_NAME}")
    run_count = 1
    solved_count = run_episodes(policy, env, NUM_SCRAMBLES)
    print(f"Initial solved_count {solved_count}")
    while solved_count < 100:

        print(f"Run count {run_count}")
        print(f"Solved_count {solved_count}")
        run_count += 1

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            max_epoch=100,
            step_per_epoch=1000,
            step_per_collect=10,
            episode_per_test=10,
            batch_size=64,
            update_per_step=0.1,
            test_in_train=False
        )

        # Save the model
        torch.save(policy.state_dict(), model_file_path)

        solved_count = run_episodes(policy, env, NUM_SCRAMBLES)
