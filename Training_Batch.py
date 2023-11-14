import os
import random
import time

from workSpace.Testing import test
from UtilityFunctions import load_model_PPO, save_model

# Hyperparameters
VERBOSE = 0

CHECKPOINT_INTERVAL = 100
MODEL_NAME = "ppo_batch_train"

if __name__ == '__main__':
    start_time = time.time()

    save_path = os.path.join('Training', 'Saved Models')

    # Create a new model by default
    training_model, env, callback = load_model_PPO(MODEL_NAME)

    # Training
    epsilon = 1.0  # Initial epsilon value
    epsilon_min = 0.01  # Minimum epsilon value
    epsilon_decay = 0.99  # Decay factor for epsilon
    TOTAL_EPISODE = 100 * 100
    for episode in range(TOTAL_EPISODE):

        # Random number of scramble for the episode
        num_scramble_list = range(1, 1 + 1)
        num_scramble = random.choice(num_scramble_list)

        # Setup to run the episode
        env.set_num_scramble(num_scramble)

        # New scramble cube
        obs = env.scramble()

        # Collect experiences
        MAX_STEPS = env.get_max_steps()
        total_num_steps = 0
        attempt = 0
        solved = False
        while not solved:
            attempt += 1

            obs, _ = env.reset()

            for step in range(MAX_STEPS):
                total_num_steps += 1

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action, _ = training_model.predict(obs)
                else:
                    action = env.action_space.sample()

                obs, reward, solved, _, _ = env.step(action)

                if solved:
                    num_steps_to_solve = step + 1
                    break

            # Greedy choice policy
            if attempt % 5 == 0:
                epsilon = max(epsilon_min, epsilon_decay * epsilon)

            if solved:
                # There is a better solution
                if env.get_current_num_steps() > num_scramble:
                    print(
                        f"Not optimal solution. Number of steps solved: {env.get_current_num_steps()}")
                    solved = False

        # Learn
        training_model.learn(total_timesteps=total_num_steps)

        # Episode stats
        print(f"Episode: {episode + 1}, number of scramble: {num_scramble}, Total steps: {total_num_steps}")

        # Checkpoint
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_model_name = MODEL_NAME + "_" + str(episode + 1)

            save_model(save_path, checkpoint_model_name, training_model)

            # Validate the model
            test(training_model, env, checkpoint_model_name)

    # Save the model
    save_model(save_path, MODEL_NAME, training_model)

    # Release resource
    env.close()

    # End time and elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Test
    test(training_model, env, MODEL_NAME)
