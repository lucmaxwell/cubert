import optuna
import torch
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer

from Network import Tianshou_Network
from RubikCubeEnv import RubikCubeEnv
from torch.optim import AdamW, Adam

from Tianshou_Model_Validation import run_episodes, episode

NUM_SCRAMBLES = 5
NUM_ENVS = 4


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suggest values for the hyperparameters to optimize
    lr = trial.suggest_categorical('lr', [2e-3, 1e-3, 2e-4, 1e-4, 2e-5])
    estimation_step = trial.suggest_categorical('estimation_step', [3, 5, 10])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # Your existing setup code for environment, model, etc.
    env = RubikCubeEnv(num_scramble=1)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Tianshou_Network(state_shape, action_shape).to(device)
    optim = Adam(net.parameters(), lr)
    policy = DQNPolicy(net, optim, estimation_step=estimation_step, is_double=True)

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

    # Your training loop, modified to use the suggested hyperparameters
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=100,
        step_per_epoch=100,
        step_per_collect=10,
        episode_per_test=10,
        batch_size=batch_size,
        update_per_step=0.1
    )

    # Return the objective value that you want to maximize or minimize
    # For example, you could return the average reward or the negative loss
    solved_count = 0
    for _ in range(100):
        done = episode(policy, env, NUM_SCRAMBLES)

        if done:
            solved_count += 1

    return solved_count


if __name__ == '__main__':
    # Running the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Set `n_trials` to the number of trials you want to run

    # Print the best hyperparameters
    print(f"Best trial: {study.best_trial.params}")
