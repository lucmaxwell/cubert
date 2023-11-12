import math
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from RubikCubeEnv import RubiksCubeEnv

# Hyperparameters
NUM_PARALLEL_ENV = 10
VERBOSE = 1

TOTAL_TIME_STEPS = 100000

NUM_SCRAMBLE = 1
MAX_STEPS_ATTEMPT = int(math.ceil(NUM_SCRAMBLE * 2.5))

MODEL_NAME = "questionable_training_ppo_1_scramble"


def make_env():
    def _init():
        env = RubiksCubeEnv()
        return env

    return _init


if __name__ == '__main__':
    start_time = time.time()

    save_path = os.path.join('Training', 'Saved Models')
    log_path = os.path.join('Training', 'Logs\\' + MODEL_NAME)
    checkpoint_path = os.path.join(save_path, 'checkpoints')

    # Create the environment
    train_envs = DummyVecEnv([make_env() for _ in range(NUM_PARALLEL_ENV)])
    eval_env = Monitor(RubiksCubeEnv())

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_path, name_prefix='rl_model_' + MODEL_NAME)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path, log_path=log_path, eval_freq=10000)
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Create or load the PPO model
    model_file_path = os.path.join(save_path, MODEL_NAME)
    if os.path.isfile(model_file_path + ".zip"):
        print("Loading existing model...")
        training_model = PPO.load(model_file_path, env=train_envs)
    else:
        training_model = PPO('MlpPolicy', env=train_envs, verbose=VERBOSE, tensorboard_log=log_path)

    # Training
    training_model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=callback_list)

    # Save the final model
    training_model.save(model_file_path)
    print(f"Model saved. Path: {model_file_path}")

    # Close the environment
    train_envs.close()
    eval_env.close()

    # End time and elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Evaluate the final model
    mean_reward, std_reward = evaluate_policy(training_model, eval_env, n_eval_episodes=100)
    print(f"Evaluation: Mean reward: {mean_reward}, Std: {std_reward}")
