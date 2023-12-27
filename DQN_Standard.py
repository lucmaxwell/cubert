import time

from Network import StandardNetwork
from UtilityFunctions import load_model_DQN, save_model
from Model_Validation import test, evaluate_model


TOTAL_STEPS = 100000
MODEL_NAME = "dqn_standard"


if __name__ == '__main__':
    start_time = time.time()

    # Create a new model by default
    training_model, env, callback = load_model_DQN(MODEL_NAME, StandardNetwork, 3)

    # Training
    training_model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

    # Save the model
    save_model(MODEL_NAME, training_model)

    # End time and elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Evaluate
    evaluate_model(training_model, env, MODEL_NAME)

    # Release resource
    env.close()
