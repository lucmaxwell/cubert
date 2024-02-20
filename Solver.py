import numpy as np
import twophase.solver as sv

class Solver:
    # Order of the cube faces in the image is B, R, F, L, U, D
    FACE_ORDER = {
        0: 'B',
        1: 'R',
        2: 'F',
        3: 'L',
        4: 'U',
        5: 'D',
    }

    def __init__(self):
        pass

    def get3x3Solution(self, imageArray, verbose=False):
        # imageToString at index i has the index of where the colour at index i belongs in the cubeString
        imageToString = np.array(
            [[45, 46, 47, 9, 10, 11, 18, 19, 20, 36, 37, 38, 2, 5, 8, 29, 32, 35],
            [48, 49, 50, 12, 13, 14, 21, 22, 23, 39, 40, 41, 1, 4, 7, 28, 31, 34],
            [51, 52, 53, 15, 16, 17, 24, 25, 26, 42, 43, 44, 0, 3, 6, 27, 30, 33]])
        
        # flipped 180 for the new camera
        # imageToString = np.array(
        #     [[53, 52, 51, 17, 16, 15, 26, 25, 24, 44, 43, 42, 6, 3, 0, 33, 30, 27],
        #      [50, 49, 48, 14, 13, 12, 23, 22, 21, 41, 40, 39, 7, 4, 1, 34, 31, 28],
        #      [47, 46, 45, 11, 10,  9, 20, 19, 18, 38, 37, 36, 8, 5, 2, 35, 32, 29]])

        # These next lines flip imageToString such that at index i, imageToString contains the index that should be at index i in cubeString 
        flat = imageToString.flatten()
        flipped = np.zeros(flat.size, dtype=np.uint32)
        for i in range(flat.size):
            flipped[i] = np.where(flat == i)[0][0]

        imageToString = flipped

        # translation stores which colour (number) corresponds to which face
        translation = np.array(['a', 'a', 'a', 'a', 'a', 'a'])
        for i in range(6):
            num = imageArray[1, 1 + 3*i]
            translation[num] = self.FACE_ORDER[i]

        imageArray = imageArray.flatten()

        cubeString = translation[imageArray[imageToString]]
        cubeString = ''.join(cubeString)

        if(verbose):
            print("Image array")
            print(imageArray)
            print()
            
            print("Ordered image array")
            print(imageArray[imageToString])
            print()
        
            print("Translation")
            print(translation)
            print()

            print("Cube string")
            print(cubeString)
            print()
            # # Pring the cube string in 
            print(f"U: {cubeString[0:3]} {cubeString[3:6]} {cubeString[6:9]}")
            print(f"R: {cubeString[9:12]} {cubeString[12:15]} {cubeString[15:18]}")
            print(f"F: {cubeString[18:21]} {cubeString[21:24]} {cubeString[24:27]}")
            print(f"D: {cubeString[27:30]} {cubeString[30:33]} {cubeString[33:36]}")
            print(f"L: {cubeString[36:39]} {cubeString[39:42]} {cubeString[42:45]}")
            print(f"B: {cubeString[45:48]} {cubeString[48:51]} {cubeString[51:54]}")
            print()

        return sv.solve(cubeString, 0, 2)

    def cubertify(self, solution, verbose=False):
        # Get the correct face facing down
        # P = yy
        # p = bb
        # D = XX

        ORIENTATE = {
            'D': "",
            'F': "X",
            'U': "D",
            'B': "PX",
            'L': "YX",
            'R': "yX"
        }

        SPINS = {
            '1': "b",
            '2': "p",
            '3': "B"
        }

        TRANSFORM = {
            'D': {
                'd': "D",
                'f': "F",
                'u': "U",
                'b': "B",
                'l': "L",
                'r': "R"
            },

            'F': {
                'd': "B",
                'f': "D",
                'u': "F",
                'b': "U",
                'l': "L",
                'r': "R"
            },

            'U': {
                'd': "U",
                'f': "B",
                'u': "D",
                'b': "F",
                'l': "L",
                'r': "R"
            },

            'B': {
                'd': "B",
                'f': "U",
                'u': "F",
                'b': "D",
                'l': "R",
                'r': "L"
            },

            'L': {
                'd': "B",
                'f': "R",
                'u': "F",
                'b': "L",
                'l': "D",
                'r': "U"
            },

            'R': {
                'd': "B",
                'f': "L",
                'u': "F",
                'b': "R",
                'l': "U",
                'r': "D"
            }
        }
        
        cuberty = ""
        move = solution[0:3]
        while move[0] != "(":

            face = move[0]
            spins = move[1]

            cuberty += ORIENTATE[face]
            cuberty += SPINS[spins]

            if(verbose):
                print(f"{move}: {ORIENTATE[face]}{SPINS[spins]}")
            
            solution = solution[3:]
            solution = solution.lower()
            for key, value in TRANSFORM[face].items():
                solution = solution.replace(key, value)

            move = solution[0:3]

        return cuberty
    
    def getMlArray(imageArray):
        # Transform from (3, 18) to (6, 3, 3)
        mlArray = np.zeros((6, 3, 3), dtype=np.uint8)
        for i in range(6):
            mlArray[i] = imageArray[:, i*3:i*3+3]

        # Put faces in the correct order
        temp = np.copy(mlArray[2])
        mlArray[2] = mlArray[0]
        mlArray[0] = temp

        # Orient faces the correct way
        mlArray[4] = np.rot90(mlArray[4], 3)
        mlArray[5] = np.rot90(mlArray[5], 3)

        return mlArray