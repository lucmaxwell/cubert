from RubikCube import RubikCube, Face

if __name__ == '__main__':
    original_obs = [[[5, 3, 2],
                     [5, 5, 4],
                     [3, 0, 3]],

                    [[3, 1, 2],
                     [2, 2, 0],
                     [0, 0, 0]],

                    [[4, 4, 5],
                     [4, 4, 2],
                     [4, 3, 1]],

                    [[1, 5, 2],
                     [3, 0, 2],
                     [2, 4, 0]],

                    [[0, 3, 1],
                     [1, 1, 2],
                     [3, 0, 4]],

                    [[4, 5, 5],
                     [1, 3, 1],
                     [5, 5, 1]]]

    # Create the environment and vector for parallel environments
    # env = RubiksCubeEnv()

    # Create the AI model
    # save_path = os.path.join('Training', 'Saved Models')
    # model_file_path = os.path.join(save_path, "AI_Model.zip")
    # model = DQN.load(model_file_path,
    #                           env=env,
    #                           verbose=2,
    #                           device="cuda")

    cube = RubikCube(3)
    cube.set_state_from_observation(original_obs)

    cube.rotate_clockwise(Face.Front)


    # Solve the cube
    # obs = env.set_observation(original_obs)
    # while not done:
    #     # Determine action and take step
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, _, done, _, _ = env.step(action)
