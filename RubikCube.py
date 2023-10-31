import random
from typing import List, Dict
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

    def set_row(self, row_list, row):
        self.square_list[row, :] = row_list

    def set_col(self, col_list, col):
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
        self.face_list = {face: RubikFace(size, self.get_default_color(face)) for face in Face}

    @staticmethod
    def get_default_color(face):
        return {
            Face.Front: SquareColour.Red,
            Face.Right: SquareColour.Green,
            Face.Back: SquareColour.Orange,
            Face.Left: SquareColour.Blue,
            Face.Top: SquareColour.Yellow,
            Face.Bottom: SquareColour.White
        }[face]

    def rotate_row_right(self, face, row):
        if face == Face.Top or face == Face.Bottom:
            if face == Face.Top:
                self.reorient(Face.Right)
            else:
                self.reorient(Face.Left)
            self.perform_rotation_col(self.size-1-row)
            self.reorient(Face.Front)
        else:
            self.perform_rotation_row(row)

    def rotate_row_left(self, face, row):
        for _ in range(3):
            self.rotate_row_right(face, row)

    def rotate_column_down(self, face, col):
        if face in [Face.Left, Face.Right, Face.Back]:
            self.reorient(face)
        self.perform_rotation_col(col)
        self.reorient(Face.Front)

    def rotate_column_up(self, face, col):
        for _ in range(3):
            self.rotate_column_down(face, col)

    def reorient(self, target_front_face):
        while self.current_front_face != target_front_face:
            for row in range(self.size):
                self.perform_rotation_row(row)
            self.current_front_face = Face((3 + self.current_front_face.value) % 4)

    def perform_rotation_row(self, row):
        horizontal_view = [Face.Front, Face.Right, Face.Back, Face.Left]
        top, bottom = Face.Top, Face.Bottom

        if row == 0:
            self.face_list[top].rotate_counter_clockwise()
        elif row == self.size - 1:
            self.face_list[bottom].rotate_clockwise()

        temp_row = self.face_list[horizontal_view[-1]].get_row(row)
        for current_face in horizontal_view:
            replace_source = temp_row
            temp_row = self.face_list[current_face].get_row(row)
            self.face_list[current_face].set_row(replace_source, row)

    def perform_rotation_col(self, col):
        vertical_view = [Face.Top, Face.Front, Face.Bottom, Face.Back]

        if col == 0:
            self.face_list[Face.Left].rotate_clockwise()
        elif col == self.size - 1:
            self.face_list[Face.Right].rotate_counter_clockwise()

        # Perform the rotation
        # Getting the back column
        # Need to do some black magic
        temp_col = np.flip(self.face_list[vertical_view[-1]].square_list[:, self.size - 1 - col])

        for current_face in vertical_view:
            # Swap the buffer column from previous
            # Buffer the next face in temp
            replace_source = temp_col
            temp_col = self.face_list[current_face].square_list[:, col].copy()

            # The index of the back face is the opposite of other face
            # The receiving has to be reserved to match the back
            if current_face == Face.Back:
                col = self.size - 1 - col
                replace_source = np.flip(replace_source)

            # Make the changes
            self.face_list[current_face].square_list[:, col] = replace_source

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
                for square in row:
                    print(square.color.ansi_color, end=" ")
                print()

        # Print the Top face
        print_face(Face.Top)

        # Print Front, Right, Back, and Left faces on the same line
        for row in range(self.size):
            for face in [Face.Front, Face.Right, Face.Back, Face.Left]:
                for square in self.face_list[face].square_list[row]:
                    print(square.color.ansi_color, end=" ")
            print()

        # Print the Bottom face
        print_face(Face.Bottom)

    def scramble(self, num_moves):
        for _ in range(num_moves):
            face = random.choice(list(Face))
            index = random.randint(0, self.size - 1)
            direction = random.choice(['up', 'down', 'left', 'right'])

            if direction == 'up':
                self.rotate_column_up(face, index)
            elif direction == 'down':
                self.rotate_column_down(face, index)
            elif direction == 'left':
                self.rotate_row_left(face, index)
            else:
                self.rotate_row_right(face, index)

    def get_face_colors(self, given_face):
        face_np = np.zeros((self.size, self.size), dtype=np.int8)
        for i in range(3):
            for j in range(3):
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


if __name__ == '__main__':
    # Initialize a 3x3 Rubik's Cube
    cube = RubikCube(3)

    # Print initial state
    print("Initial cube state:")
    cube.print_cube_state()

    # Test 1: Confirm that the cube starts in a solved state
    assert cube.is_solved(), "Failed Test 1: Initial cube should be solved"

    # Test 2: Perform some rotations and confirm it's not solved

    cube.rotate_row_right(Face.Front, 0)
    print("Front row 0:")
    cube.print_cube_state()

    cube.rotate_column_down(Face.Right, 1)
    print("Right col 1:")
    cube.print_cube_state()

    cube.rotate_row_left(Face.Back, 2)
    print("Back row 2:")
    cube.print_cube_state()

    cube.rotate_row_left(Face.Left, 0)
    print("Left row 0:")
    cube.print_cube_state()

    cube.rotate_column_down(Face.Back, 0)
    print("Back col 0:")
    cube.print_cube_state()


    assert not cube.is_solved(), "Failed Test 2: Cube should not be solved"

    # Test 3: Undo the rotations and confirm it's solved

    cube.rotate_column_up(Face.Back, 0)
    print("Reverse Back col 0:")
    cube.print_cube_state()

    cube.rotate_row_right(Face.Left, 0)
    print("Reverse Left row 0:")
    cube.print_cube_state()

    cube.rotate_row_right(Face.Back, 2)
    print("Reverse Back row 2:")
    cube.print_cube_state()

    cube.rotate_column_up(Face.Right, 1)
    print("Reverse Right col 1:")
    cube.print_cube_state()

    cube.rotate_row_left(Face.Front, 0)
    print("Reverse Front row 0:")
    cube.print_cube_state()

    assert cube.is_solved(), "Failed Test 3: Cube should be solved after undoing the rotations"

    # Test 4: Perform 30 random moves and then undo them
    moves = []
    for _ in range(30):
        face = random.choice(list(Face))
        index = random.randint(0, cube.size - 1)
        direction = random.choice(['up', 'down', 'left', 'right'])

        if direction == 'up':
            cube.rotate_column_up(face, index)
        elif direction == 'down':
            cube.rotate_column_down(face, index)
        elif direction == 'left':
            cube.rotate_row_left(face, index)
        else:
            cube.rotate_row_right(face, index)

        moves.append((face, index, direction))

    assert not cube.is_solved(), "Failed Test 4: Cube should not be solved after 30 random moves"

    # Undo the 30 moves
    for face, index, direction in reversed(moves):
        if direction == 'up':
            cube.rotate_column_down(face, index)
        elif direction == 'down':
            cube.rotate_column_up(face, index)
        elif direction == 'left':
            cube.rotate_row_right(face, index)
        else:
            cube.rotate_row_left(face, index)

    assert cube.is_solved(), "Failed Test 4: Cube should be solved after undoing the 30 moves"

    # Test 5: Scramble and print the final state
    cube.scramble(100)
    print("Scrambled cube:")
    cube.print_cube_state()

    assert not cube.is_solved(), "Failed Test 5: Scramble cube is not solved"

    print("All tests passed!")
