import os

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from torch.optim import Adam

from Network import Tianshou_Network
from RubikCubeEnv import RubikCubeEnv, decode_action

if __name__ == '__main__':

    obs_target = np.array(
        [[[3, 0, 2],
          [4, 0, 2],
          [0, 0, 0]],

         [[1, 1, 0],
          [1, 1, 0],
          [5, 5, 4]],

         [[5, 2, 1],
          [5, 2, 4],
          [1, 4, 4]],

         [[2, 3, 4],
          [3, 3, 1],
          [3, 3, 3]],

         [[4, 4, 1],
          [0, 4, 5],
          [0, 1, 5]],

         [[2, 2, 3],
          [2, 5, 3],
          [2, 5, 5]]]
                        )

    MODEL_NAME = "DQN_Tianshou_Vector_5"

    save_path = os.path.join('Training', 'Saved Models')

    # Set up the environment
    env = RubikCubeEnv(num_scramble=0)

    # Set up the network and policy
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Tianshou_Network(state_shape, action_shape)

    # Parameters
    optim = Adam(net.parameters(), lr=2e-4)
    policy = DQNPolicy(
        net,
        optim,
        estimation_step=10
    )

    # Load progress
    model_file_path = os.path.join(save_path, MODEL_NAME + '.pth')
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        policy.load_state_dict(torch.load(model_file_path))

        net.eval()

        obs = env.set_observation(obs_target)
        print("Original:")
        env.render()

        # Solve
        done = False
        count = 0
        while not done and count < 10:
            count += 1

            batch = Batch(obs=np.array([obs]), info={})
            action = policy(batch).act[0]
            obs, _, done, _, _ = env.step(action)

            print(f"Count {count}")
            print(f"Decode action {decode_action(action)}")
            env.render()

        done = env.is_solved()

        print(f"Solved {done}")
