import os
import time
import matplotlib.pyplot as plt

from Model_Validation import evaluate_scramble_1000, evaluate_model


def train_and_evaluate(training_model, save_path, MODEL_NAME, NUM_STEPS, NUM_SCRAMBLES, plot_folder_name):
    # Initialize training variables
    run_count = 0
    done = False

    # Start the training loop
    while not done:
        run_count += 1
        start_time = time.time()

        # Training
        training_model.learn(total_timesteps=NUM_STEPS)

        # Save the model
        model_file_path = os.path.join(save_path, MODEL_NAME + ".zip")
        training_model.save(model_file_path)
        print(f"Model {MODEL_NAME} saved. Path: {model_file_path}")

        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time {run_count}: {elapsed_time} seconds")

        # Check if 100% solve rate condition is met
        done, count, solved_percentage = evaluate_scramble_1000(training_model, NUM_SCRAMBLES)
        print(f"{run_count} num_scramble={NUM_SCRAMBLES}: {count} {solved_percentage}%")

    # Plot the model evaluation
    plot_evaluation(training_model, MODEL_NAME, NUM_SCRAMBLES, run_count, plot_folder_name)


def plot_evaluation(training_model, MODEL_NAME, NUM_SCRAMBLES, run_count, plot_folder_name):
    # Get evaluation results
    num_scrambles, evaluation_results = evaluate_model(training_model)
    plot_title = f"{MODEL_NAME} scramble={NUM_SCRAMBLES} run_count={run_count}"
    plt.figure(figsize=(10, 6))
    plt.plot(num_scrambles, evaluation_results, marker='o')
    plt.title(plot_title)
    plt.xlabel('Number of Scrambles')
    plt.ylabel('Solved Counts')
    plt.xticks(num_scrambles)
    plt.grid(True)

    # Annotate each point with its value
    for i, count in enumerate(evaluation_results):
        plt.annotate(str(count), (num_scrambles[i], evaluation_results[i]), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    # Save the plot
    plt.savefig(os.path.join(plot_folder_name, f'{plot_title}.png'))

    # Show the plot
    plt.show()
