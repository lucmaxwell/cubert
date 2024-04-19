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
                              [5, 5, 2],]])

    env = RubikCubeEnv(num_scramble=0)
    found = False
    count = 0
    while not found and count < 1_000_000:
        count += 1
        obs = env.scramble(3)

        if np.array_equal(obs_target, obs):
            found = True

    print(f"Found {count} {found}")
