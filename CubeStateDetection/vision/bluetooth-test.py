import socket
import msvcrt
# import random
import cv2 as cv
import numpy as np
import os
# import glob
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import urllib.request
# import scipy.stats as stats
import time
import vision
import solver

y = b'y'
y_ = b'Y'
b = b'b'
b_ = b'B' 
x_ = b'X'
ACK = b'a'
OK = b'k'

basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"
imagesPath = basePath + "images/"
imageUrl = "http://192.168.4.1/capture"
mask = 'mask2.png'
maskPath = imagesPath + mask

def getAllImages(imageUrl, client, maskPath=""):
        # left is from left of image, top is from top of image, height = width
        top = 102
        left = 250
        height = 339

        if(maskPath != ""):
            imageMask = vision.loadMask(maskPath)
        else:
            imageMask = vision.getBlankMask(height)

        combinedImage = np.zeros((height, height * 6, 3), np.uint8)
        combinedMask = np.zeros((height, height * 6, 3), np.uint8)

        client.setblocking(True)

        for i in range(6):
            img = vision.getImage(imageUrl)
            img = vision.crop(img, top, left, height)

            if i == 0 or i == 1 or i == 2:
                client.send(x_)
                expectedAcks = 1
            elif i == 3:
                client.send(y)
                client.send(x_)
                client.send(y_)
                expectedAcks = 3
            elif i == 4:
                client.send(x_)
                client.send(x_)
                expectedAcks = 2
            else:
                expectedAcks = 0

            combinedImage[0:height, i*height:(i+1)*height, 0:3] = img
            combinedMask[0:height, i*height:(i+1)*height, 0:3] = imageMask
            
            data = ''
            ackCount = 0
            while ackCount < expectedAcks:
                data = client.recv(1)
                if(data == b'a'):
                    ackCount += 1
                    print(f'ack {ackCount}/{expectedAcks} received')

            print("Cube rotated, waiting 2 seconds for camera to stabilize")
            if(i != 5):
                time.sleep(2)

        client.setblocking(False)

        return combinedImage, combinedMask

def send(client, message):
    client.send(message.encode("utf-8"))
    print(f"Sent '{message}'")

def solve(maskPath="", loadFromDisk=False, writeImages=False):
    
    imageName = "0testing.png"
    maskName = "0testingMask.png"

    print(f"Starting cube solving sequence")

    # Take images
    print("Taking images")
    if( not loadFromDisk):
        cube, mask = getAllImages(imageUrl, client, maskPath)
    else:
        cube = vision.loadCube(imagesPath + imageName)
        mask = vision.loadMask(imagesPath + maskName)

    # Write output images
    if(writeImages):
        cv.imwrite(imagesPath + imageName, cube)
        cv.imwrite(imagesPath + maskName, mask * 255)

    print("Images taken")
    print()
    
    # Find cube state
    print("Finding cube state")
    cubeState = vision.getCubeState(cube, mask, 3, 18, True)
    print("Got cube state")
    print(cubeState)
    print()

    # Find cube solution
    print("Finding solution")
    solution = solver.get3x3Solution(cubeState)
    print("Found solution")
    print(solution)
    print()

    # Abort if the solver had an error
    if(solution.startswith("Error: ")):
        print("Aborting solution attempt")
        return
    
    # Translate solution
    print("Translating to cubertish")
    cubertSolution = solver.cubertify(solution)
    print("Translated to cubertish")
    print(cubertSolution)
    print()

    # Send instructions
    print("Sending instructions")
    send(client, cubertSolution)
    print("Instructions sent")

client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
# client.connect(("40:22:D8:F0:E6:1A", 1))    # ESP32test
client.connect(("40:22:D8:EB:2B:3A", 1))    # ESP32test2
client.setblocking(False)
print("Connected to ESP32 through bluetooth")

message = ''

while True:
    if msvcrt.kbhit():
        letter = msvcrt.getch().decode("utf-8")

        if letter == "\r":
            print()
            match message:
                case "solve":
                    solve(maskPath)

                case "disk":
                    solve(maskPath, True, True)

                case _:
                    send(client, message)
                    message = ""
            
        else: 
            message += letter
            print(letter, end="")

    try:
        data = client.recv(1)

    except:
        print("",end="")