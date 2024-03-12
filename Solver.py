import numpy as np
import twophase.solver as sv
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from torch.optim import AdamW
from torch import load

from RubikCubeEnv import RubikCubeEnv, decode_action
from machine_learning.Network import Tianshou_Network


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
        self.policy = None
        self.environment = None

    def __del__(self):
        print("Deleting Solver")

    def get3x3Solution(self, imageArray, verbose=False):
        # imageToString at index i has the index of where the colour at index i belongs in the cubeString
        imageToString = np.array(
            [[45, 46, 47, 9, 10, 11, 18, 19, 20, 36, 37, 38, 2, 5, 8, 29, 32, 35],
            [48, 49, 50, 12, 13, 14, 21, 22, 23, 39, 40, 41, 1, 4, 7, 28, 31, 34],
            [51, 52, 53, 15, 16, 17, 24, 25, 26, 42, 43, 44, 0, 3, 6, 27, 30, 33]])
        
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
    
    def getMlArray(self, imageArray):
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


        # Translate the values such that face 0's centre is 0, face 1's centre is 1, and so on 
        mlArray = mlArray + 1
        mlArray = mlArray * 10
        for i in range(6):
            mlArray[mlArray == mlArray[i, 1, 1]] = i

        return mlArray

    def getAiSolution(self, cubeState, verbose=True):

        if (self.environment == None or self.policy == None):
            self.loadModel()

        else:
            print("Model Already Loaded. Finding Solution")

        self.environment.set_observation(cubeState)
        
        if(verbose):
            print("AI Cube state (model):")
            self.environment.render()

        # Moves
        print("Solve...")
        action_list = []
        done = False
        attempt_count = 0
        while not done and attempt_count < 3:
            attempt_count += 1
            action_list = []

            obs = self.environment.set_observation(cubeState)

            move_count = 0
            while not done and move_count < 5:
                move_count += 1

                batch = Batch(obs=np.array([obs]), info={})
                action = self.policy(batch).act[0]
                obs, _, done, _, _ = self.environment.step(action)

                action_list.append(action)

                if (verbose):
                    face, spin = decode_action(action)
                    print(f"{attempt_count} {move_count} Action: {face} {spin}")
                    self.environment.render()

            done = self.environment.is_solved()

        # Was not able to solve
        if not done:
            action_list = []

        if verbose:
            print("AI final cube state:")
            self.environment.render()
            print(action_list)

        # Return
        return action_list

    def loadModel(self):
        # Setup the machine learning model
        print("Setup machine learning model...")
        self.environment = RubikCubeEnv()
        state_shape = self.environment.observation_space.shape or self.environment.observation_space.n
        action_shape = self.environment.action_space.shape or self.environment.action_space.n
        network = Tianshou_Network(state_shape, action_shape)
        optim = AdamW(network.parameters(), lr=1e-3)
        self.policy = DQNPolicy(network, optim, estimation_step=10)

        # Load the saved policy state
        print("Load network...")
        MODEL_NAME = "DQN_Tianshou_Vector.pth"
        model_path = '/home/pi/cubert/machine_learning/' + MODEL_NAME
        self.policy.load_state_dict(load(model_path, map_location='cpu'))
        network.eval()  # Set to eval mode

if __name__ == '__main__':
    print("Starting main")    
    solver = Solver()
    a = solver.getAiSolution(np.array([1, 2, 3, 4]))
    print("AI solution:")
    print(a)