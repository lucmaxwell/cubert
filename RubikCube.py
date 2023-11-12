import random
from enum import Enum
import copy

import numpy as np


class Face(Enum):
    Front = 0
    Right = 1
    Back = 2
    Left = 3
    Top = 4
    Bottom = 5


class SquareColour(Enum):
    Red = 0
    Green = 1
    Orange = 2
    Blue = 3
    Yellow = 4
    White = 5

    @property
    def ansi_color(self):
        color_map = {
            SquareColour.Red: "\033[91m█\033[0m",
            SquareColour.Green: "\033[92m█\033[0m",
            SquareColour.Orange: "\033[38;5;208m█\033[0m",
            SquareColour.Blue: "\033[94m█\033[0m",
            SquareColour.Yellow: "\033[93m█\033[0m",
            SquareColour.White: "\033[97m█\033[0m"
        }
        return color_map[self]


class Square:
    def __init__(self, color):
        self.color = color

class RubikFace:
    def __init__(self, size, color):
        self.size = size
        self.square_list = np.array([[Square(color) for _ in range(size)] for _ in range(size)], dtype=object)

    def get_row(self, row):
        return copy.deepcopy(self.square_list[row, :])

    def get_col(self, col):
        return copy.deepcopy(self.square_list[:, col])

    def set_row(self, row, row_list):
        self.square_list[row, :] = row_list

    def set_col(self, col, col_list):
        self.square_list[:, col] = col_list

    def rotate_clockwise(self):
        self.square_list = np.rot90(self.square_list, -1)

    def rotate_counter_clockwise(self):
        self.square_list = np.rot90(self.square_list, 1)

    def get_square_list(self):
        return np.array([[copy.deepcopy(square) for square in row] for row in self.square_list], dtype=object)

class RubikCube:
    def __init__(self, size):
        self.size = size
        self.current_front_face = Face.Front
        self.face_list = {}
        self.reset()

    def reset(self):
        self.current_front_face = Face.Front
        self.face_list = {face: RubikFace(self.size, self.get_default_color(face)) for face in Face}

    @staticmethod
    def get_default_color(given_face):
        return {
            Face.Front: SquareColour.Red,
            Face.Right: SquareColour.Green,
            Face.Back: SquareColour.Orange,
            Face.Left: SquareColour.Blue,
            Face.Top: SquareColour.Yellow,
            Face.Bottom: SquareColour.White
        }[given_face]

    def rotate_clockwise(self, face):
        self._perform_clockwise_rotation(face)

    def rotate_counter_clockwise(self, face):
        for _ in range(3):
            self._perform_clockwise_rotation(face)

    def _perform_clockwise_rotation(self, face):
        if (face is Face.Top) or (face is Face.Bottom):
            rotate_row = 0
            face_list = [Face.Front, Face.Left, Face.Back, Face.Right]

            # The rotation face is the bottom
            if face is Face.Bottom:
                self.face_list[Face.Bottom].rotate_clockwise()
                face_list.reverse()
                rotate_row = self.size - 1
            else:
                self.face_list[Face.Top].rotate_clockwise()

            # Rotate the first row across the adjacent faces
            from_row = self.face_list[face_list[0]].get_row(rotate_row)
            for i in range(len(face_list)):
                # Face the rotate to state
                to_face = face_list[(i + 1) % len(face_list)]
                temp = self.face_list[to_face].get_row(rotate_row)

                # Perform the rotation
                self.face_list[to_face].set_row(rotate_row, from_row)

                # For the next rotation
                from_row = temp

        else:
            # The give face is front
            if face is Face.Front:
                left_to_top = self.face_list[Face.Left].get_col(self.size - 1)[::-1]
                top_to_right = self.face_list[Face.Top].get_row(self.size - 1)
                right_to_bottom = self.face_list[Face.Right].get_col(0)[::-1]
                bottom_to_left = self.face_list[Face.Bottom].get_row(0)

                self.face_list[Face.Left].set_col(self.size - 1, bottom_to_left)
                self.face_list[Face.Top].set_row(self.size - 1, left_to_top)
                self.face_list[Face.Right].set_col(0, top_to_right)
                self.face_list[Face.Bottom].set_row(0, right_to_bottom)

            # The given face is left
            if face is Face.Left:
                left_to_top = self.face_list[Face.Back].get_col(self.size - 1)[::-1]
                top_to_right = self.face_list[Face.Top].get_col(0)
                right_to_bottom = self.face_list[Face.Front].get_col(0)
                bottom_to_left = self.face_list[Face.Bottom].get_col(0)[::-1]

                self.face_list[Face.Back].set_col(self.size - 1, bottom_to_left)
                self.face_list[Face.Top].set_col(0, left_to_top)
                self.face_list[Face.Front].set_col(0, top_to_right)
                self.face_list[Face.Bottom].set_col(0, right_to_bottom)

            # The given face is right
            elif face is Face.Right:
                left_to_top = self.face_list[Face.Front].get_col(self.size - 1)
                top_to_right = self.face_list[Face.Top].get_col(self.size - 1)[::-1]
                right_to_bottom = self.face_list[Face.Back].get_col(0)[::-1]
                bottom_to_left = self.face_list[Face.Bottom].get_col(self.size - 1)

                self.face_list[Face.Front].set_col(self.size - 1, bottom_to_left)
                self.face_list[Face.Top].set_col(self.size - 1, left_to_top)
                self.face_list[Face.Back].set_col(0, top_to_right)
                self.face_list[Face.Bottom].set_col(self.size - 1, right_to_bottom)

            # The given face is back
            elif face is Face.Back:
                left_to_top = self.face_list[Face.Right].get_col(self.size - 1)
                top_to_right = self.face_list[Face.Top].get_row(0)[::-1]
                right_to_bottom = self.face_list[Face.Left].get_col(0)
                bottom_to_left = self.face_list[Face.Bottom].get_row(self.size - 1)[::-1]

                self.face_list[Face.Right].set_col(self.size - 1, bottom_to_left)
                self.face_list[Face.Top].set_row(0, left_to_top)
                self.face_list[Face.Left].set_col(0, top_to_right)
                self.face_list[Face.Bottom].set_row(self.size - 1, right_to_bottom)

    def is_solved(self) -> bool:
        for face, rubik_face in self.face_list.items():
            first_square_color = rubik_face.square_list[0][0].color
            for row in rubik_face.square_list:
                for square in row:
                    if square.color != first_square_color:
                        return False
        return True

    def print_cube_state(self):
        # Helper function to print a face
        def print_face(face):
            for row in self.face_list[face].square_list:
                print("      ", end="")  # Further adjusted indentation for top and bottom faces
                for square in row:
                    print(square.color.ansi_color, end=" ")
                print()

        # Print the Top face
        print_face(Face.Top)

        # Print Front, Right, Back, and Left faces on the same line
        for row in range(self.size):
            for face in [Face.Left, Face.Front, Face.Right, Face.Back]:
                for square in self.face_list[face].square_list[row]:
                    print(square.color.ansi_color, end=" ")
            print()

        # Print the Bottom face
        print_face(Face.Bottom)

    def scramble(self, moves=13):
        """
        Scrambles the cube by performing a series of random rotations,
        ensuring that consecutive moves do not undo each other.

        Parameters:
        moves (int): The number of random moves to perform.
        """
        last_move = None

        def opposite_direction(self, direction):
            """
            Returns the opposite direction for a given direction.
            """
            return 'counter_clockwise' if direction == 'clockwise' else 'clockwise'

        for _ in range(moves):
            while True:
                # Choose a random face and direction
                face = random.choice(list(Face))
                direction = random.choice(['clockwise', 'counter_clockwise'])

                # Check if the current move is the inverse of the last move
                if last_move is None or (last_move != (face, self.opposite_direction(direction))):
                    break

            # Perform the rotation
            if direction == 'clockwise':
                self.rotate_clockwise(face)
            else:
                self.rotate_counter_clockwise(face)

            # Update last move
            last_move = (face, direction)

    def get_face_colors(self, given_face):
        face_np = np.zeros((self.size, self.size), dtype=np.uint8)
        for i in range(self.size):
            for j in range(self.size):
                face_np[i, j] = self.face_list[given_face].square_list[i][j].color.value
        return face_np

    def count_correct_squares(self):
        correct_count = 0
        for face_element, rubik_face in self.face_list.items():
            correct_color = self.get_default_color(face_element)
            for row in rubik_face.square_list:
                for square in row:
                    if square.color == correct_color:
                        correct_count += 1
        return correct_count

    def set_state_from_observation(self, observation):
        """Initialize the state of the cube based on the given observation."""
        for face_index, face in enumerate(Face):
            for i in range(self.size):
                for j in range(self.size):
                    color_value = observation[face_index][i][j]
                    self.face_list[face].square_list[i][j].color = SquareColour(color_value)


if __name__ == '__main__':
    # Initialize a 3x3 Rubik's Cube
    cube = RubikCube(3)

    # Print initial state
    print("Initial cube state:")
    cube.print_cube_state()

    # Test 1: Confirm that the cube starts in a solved state
    assert cube.is_solved(), "Failed Test 1: Initial cube should be solved"

    # Print the individual rotation
    cube = RubikCube(3)
    cube.rotate_clockwise(Face.Front)
    print("Front clockwise:")
    cube.print_cube_state()

    cube = RubikCube(3)
    cube.rotate_clockwise(Face.Right)
    print("Right clockwise:")
    cube.print_cube_state()

    cube = RubikCube(3)
    cube.rotate_clockwise(Face.Left)
    print("Left clockwise:")
    cube.print_cube_state()

    cube = RubikCube(3)
    cube.rotate_clockwise(Face.Back)
    print("Back clockwise:")
    cube.print_cube_state()

    cube = RubikCube(3)
    cube.rotate_clockwise(Face.Top)
    print("Top clockwise:")
    cube.print_cube_state()

    cube = RubikCube(3)
    cube.rotate_clockwise(Face.Bottom)
    print("Bottom clockwise:")
    cube.print_cube_state()

    # Scramble the cube with 30 random moves and record the moves
    cube = RubikCube(3)
    scramble_moves = []
    for _ in range(30):
        face = random.choice(list(Face))
        direction = random.choice(['clockwise', 'counter_clockwise'])
        scramble_moves.append((face, direction))
        if direction == 'clockwise':
            cube.rotate_clockwise(face)
        else:
            cube.rotate_counter_clockwise(face)

    # Print scrambled state
    print("Scrambled cube:")
    cube.print_cube_state()

    # Test 2: Check that the cube is not solved
    assert not cube.is_solved(), "Failed Test 2: Cube should not be solved after scrambling"

    # Undo the scramble by reversing the moves
    for face, direction in reversed(scramble_moves):
        if direction == 'clockwise':
            cube.rotate_counter_clockwise(face)
        else:
            cube.rotate_clockwise(face)

    # Print the state after undoing the scramble
    print("Cube after undoing the scramble:")
    cube.print_cube_state()

    # Test 3: Check that the cube is solved after undoing the scramble
    assert cube.is_solved(), "Failed Test 3: Cube should be solved after undoing the scramble"

    print("All tests passed!")
