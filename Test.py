from RubikCubeEnv import RubiksCubeEnv

if __name__ == '__main__':
    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()

    # Create the AI model
    model = DQN.load(model_file_path,
                              env=env,
                              verbose=2,
                              device="cuda")

