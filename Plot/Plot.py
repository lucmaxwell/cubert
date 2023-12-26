import pandas as pd
import matplotlib.pyplot as plt


def plot(file_paths, title):
    # Reading the CSV files into pandas dataframes
    dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

    # Create a set to store unique num_scramble values
    num_scrambles = set()

    # Create figure before plotting
    plt.figure(figsize=(10, 6))

    # Plot
    for name, df in dataframes.items():
        plt.plot(df['num_scramble'], df['solved_percentage'], label=name)
        num_scrambles.update(df['num_scramble'].unique())

    # Labels
    plt.xlabel('Number of Scrambles')
    plt.ylabel('Solved Percentage')
    plt.title(title)
    plt.legend()

    # Set x-ticks to unique num_scramble values
    plt.xticks(sorted(num_scrambles))

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # File paths
    file_paths = {
        'Standard': './dqn_standard.csv',
        'Ascending': './dqn_ascending.csv',
    }

    plot(file_paths, '1,000,000 steps, 3 scramble')
