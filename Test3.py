import numpy as np

from RubikCube import RubikCube
from RubikCubeEnv import RubikCubeEnv

if __name__ == '__main__':
    obs_target = np.array([[[0, 0, 1],
                  [0, 0, 5],
                  [0, 0, 5],],

                 [[4, 4, 4],
                  [1, 1, 1],
                  [1, 1, 1],],

                 [[3, 2, 2],
                  [4, 2, 2],
                  [4, 2, 2],],

                 [[3, 3, 5],
                  [3, 3, 5],
                  [3, 3, 5],],

                 [[4, 4, 0],
                  [4, 4, 0],
                  [3, 3, 0],],

                 [[1, 1, 2],
                  [5, 5, 2],
                  [5, 5, 2],],])

    print("Target obs:")
    print(obs_target)
    print()

    env = RubikCubeEnv(num_scramble=0)
    found = False
    count = 0

    face = 0
    spin = 0
    action = face * 2 + spin
    obs, _, _, _, _ = env.step(action)

    print(f"Action {face} {spin}")
    env.render()
    print(f"Obs: {obs}")

    face = 1
    spin = 0
    action = face * 2 + spin
    obs, _, _, _, _ = env.step(action)

    print(f"Action {face} {spin}")
    env.render()
    print(f"Obs: {obs}")

    print(f"Same {np.array_equal(obs_target, obs)}")
