import os

import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from RubikCubeEnv import RubiksCubeEnv

save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')

def load_model_PPO(model_name, num_scramble=1):
    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv(num_scramble=num_scramble)

    # Create a new model by default
    training_model = None
    model_file_path = os.path.join(save_path, model_name + ".zip")
    model_log_path = os.path.join(log_path, '\\' + model_name)
    if os.path.isfile(model_file_path):
        print(f"Loading existing model from {model_file_path}")
        training_model = PPO.load(model_file_path, env=env, verbose=2, tensorboard_log=model_log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = PPO('MlpPolicy', env=env, verbose=2, tensorboard_log=model_log_path, device="cuda")

    # Callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=os.path.join(save_path, model_name),
                                             name_prefix=model_name)
    eval_env = Monitor(RubiksCubeEnv(num_scramble=num_scramble))
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(save_path, model_name + "_best_model"),
                                 log_path=os.path.join(model_log_path, model_name + "_best_model"), eval_freq=10000,
                                 deterministic=True)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Return
    return training_model, env, callback


def load_model_DQN(model_name):
    print(torch.__version__)
    print(f"CUDA is available: {torch.cuda.is_available()}")

    # Create the environment and vector for parallel environments
    env = RubiksCubeEnv()

    # Create a new model by default
    training_model = None
    model_file_path = os.path.join(save_path, model_name + ".zip")
    model_log_path = os.path.join(log_path, '\\' + model_name)
    if os.path.isfile(model_file_path):
        print("Loading existing model...")
        training_model = DQN.load(model_file_path, env=env, verbose=2,tensorboard_log=model_log_path, device="cuda")
    else:
        print("Creating a new model...")
        training_model = DQN('MlpPolicy', env=env, verbose=2, tensorboard_log=model_log_path, device="cuda")

    # Callback
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=os.path.join(save_path, model_name),
                                             name_prefix=model_name)
    eval_env = RubiksCubeEnv()
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(save_path, model_name + "_best_model"),
                                 log_path=os.path.join(model_log_path, "_best_model"), eval_freq=500,
                                 deterministic=True)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Return
    return training_model, env, callback


def save_model(model_name, model):
    model_file_path = os.path.join(save_path, model_name + ".zip")

    model.save(model_file_path)
    print(f"Model {model_name} saved. Path: {model_file_path}")
