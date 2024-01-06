import os

import torch
from stable_baselines3 import DQN

from Network import ResidualBlock_2Layers_2048, ResidualBlock_3Layers_2048, ResidualBlock_2Layers_4096
from RubikCubeStaticEnv import RubiksCubeStaticEnv
from Training_Utility_Functions import train_and_evaluate

network_configuration = ResidualBlock_2Layers_2048

NUM_SCRAMBLES = 4

NUM_EPISODES = 10_000


if __name__ == '__main__':
    MODEL_NAME = "DQN_Static_Env_" + network_configuration.__name__

    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')

    # Ensure the directory exists
    plot_folder_name = 'training_plot'
    os.makedirs(plot_folder_name, exist_ok=True)

    # Create the environment and vector for parallel environments
    env = RubiksCubeStaticEnv(num_scramble=NUM_SCRAMBLES)

    # Define the policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=network_configuration,
        features_extractor_kwargs=dict(features_dim=env.action_space.n)
    )

    # Create a new model or load model if already existed
    training_model = None
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = DQN.load(model_file_path,
                                  env=env,
                                  verbose=0,
                                  device="cuda",
                                  learning_starts=1_000)
    else:
        print("Creating a new model...")
        training_model = DQN(policy='MlpPolicy',
                             env=env,
                             policy_kwargs=policy_kwargs,
                             verbose=0,
                             device="cuda",
                             learning_starts=1_000)

    # Learn and evaluate
    train_and_evaluate(training_model, save_path, MODEL_NAME, NUM_EPISODES, NUM_SCRAMBLES, plot_folder_name)

    # Release resource
    env.close()
