import os

import torch
from stable_baselines3 import DQN

from RubikCubeEnv import RubiksCubeEnv
from Training_Utility_Functions import train_and_evaluate

NUM_SCRAMBLES = 3

NUM_STEPS = 100_000


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    MODEL_NAME = "DQN_No_Network_Configuration"

    print(f"Model name: {MODEL_NAME}")

    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    save_path = os.path.join('Training', 'Saved Models')

    # Ensure the directory exists
    plot_folder_name = 'training_plot'
    os.makedirs(plot_folder_name, exist_ok=True)

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv(num_scramble=NUM_SCRAMBLES)

    # Create a new model or load model if already existed
    training_model = None
    model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = DQN.load(model_file_path,
                                  env=env,
                                  verbose=0,
                                  device="cuda")
    else:
        print("Creating a new model...")
        training_model = DQN(policy='MlpPolicy',
                             env=env,
                             verbose=0,
                             device="cuda")

    # Learn and evaluate
    train_and_evaluate(
        training_model,
        save_path,
        MODEL_NAME,
        NUM_STEPS,
        NUM_SCRAMBLES,
        plot_folder_name
    )

    # Release resource
    env.close()
