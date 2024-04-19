from matplotlib import pyplot as plt
from RubikCube import Face, RubikCube

if __name__ == '__main__':
    # Define all moves
    all_moves = []
    for face in Face:
        for direction in ['clockwise', 'counter_clockwise']:
            all_moves.append((face, direction))

    # Initialize rewards list
    rewards = []

    # Original cube state
    cube = RubikCube(3)
    cube.rotate_clockwise(Face.Bottom)
    obs = cube.get_observation()

    for move in all_moves:
        # Reinitialize the cube to its solved state for each move
        cube.set_state_from_observation(obs)

        # Perform the move
        face, direction = move
        if direction == 'clockwise':
            cube.rotate_clockwise(face)
        else:
            cube.rotate_counter_clockwise(face)

        # Calculate the reward, fixing the reward calculation
        done = cube.is_solved()  # This will always be False here
        #reward = 1 if done else -0.5 - 0.5 * (1 - cube.percentage_correct())
        reward = 1 if done else -1

        # Collect the reward
        rewards.append(reward)

    # Plotting
    plt.figure(figsize=(12, 6))  # Adjust size as needed
    move_labels = [f"{face.name} {dir}" for face, dir in all_moves]
    plt.stem(move_labels, rewards)  # Use line collections for better performance
    plt.title('Reward Impact for Specific Moves on an 1 scrambled Rubik\'s Cube')
    plt.xlabel('Move')
    plt.ylabel('Reward')
    # Increase font size for x-axis labels
    plt.xticks(rotation=45, ha="right", fontsize=15)  # Adjust the fontsize as needed
    plt.ylim(-1, 1)  # Set the y-axis to span from -1 to 1
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.grid(axis='y')
    plt.show()
