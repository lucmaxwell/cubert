import socket
import msvcrt
import random
import cv2 as cv
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import urllib.request
import scipy.stats as stats
import time
import vision
import solver

# def getImage(url):
#     req = urllib.request.urlopen(url)
#     arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#     img = cv.imdecode(arr, -1) # 'Load it as it is'
#     return img

def getAllImages(url, client, imageName, maskName, useMask):
        # left is from left of image, top is from top of image, height = width
        top = 102
        left = 250
        height = 339
        width = height

        if(useMask):
            imageMask = cv.imread(maskPath)
            imageMask = imageMask.astype(np.uint8)
            imageMask[imageMask != 255] = 0
            imageMask[imageMask == 255] = 255
        
        combinedImage = np.zeros((height, height * 6, 3), np.uint8)
        combinedMask = np.zeros((height, height * 6, 3), np.uint8)

        for i in range(6):
            img = vision.getImage(imageUrl)
            img = img[top:top+height, left:left+width, :]

            if i == 0 or i == 1 or i == 2:
                client.send(x_)
            elif i == 3:
                client.send(y)
                client.send(x_)
                client.send(y_)
            elif i == 4:
                client.send(x_)
                client.send(x_)

            combinedImage[0:height, i*height:(i+1)*height, 0:3] = img
            combinedMask[0:height, i*height:(i+1)*height, 0:3] = imageMask
            time.sleep(5)

        cv.imwrite(imagesPath + imageName, combinedImage)
        cv.imwrite(imagesPath + maskName, combinedMask)

# Parameters
useUrl = True
clearOutputDirectory = False
image = "test2 modified.jpg"
mask = 'mask.png'
edgeLength = 3
edgeHeight =3
numColours = 6

useMask = True
useAutoMask = False
maskMin = 50
maskMax = 230

# Kind of also parameters but not really
basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"
imagesPath = basePath + "images/"
outPath = basePath + "output/"
imageUrl = "http://192.168.4.1/capture"
cube = imagesPath + image
maskPath = imagesPath + mask

y = b'y'
y_ = b'Y'
b = b'b'
b_ = b'B' 
x_ = b'X'
ACK = b'a'
OK = b'k'

messages = [y, y_, b, b_, x_]

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
                case "start":
                    print(f"Starting imaging sequence")
                    
                    imageName = "0testing.png"
                    maskName = "0testingMask.png"

                    print("Taking images")
                    test = getAllImages(imageUrl, client, imageName, maskName, True)
                    print("Images taken")
                    print("Finding cube state")
                    cubeState = vision.getCubeState(imageName, maskName, True)
                    print("Got cube state")
                    print(cubeState)
                    print()

                    print("Finding solution")
                    solution = solver.get3x3Solution(cubeState)
                    print("Found solution")
                    print(solution)
                    print()

                    print("Translating to cubertish")
                    cubertSolution = solver.cubertify(solution)
                    print("Translated to cubertish")
                    print(cubertSolution)

                    print("Sending instructions")
                    client.send(cubertSolution.encode("utf-8"))
                    print("Instructions sent")
                    message = ""

                case _:
                    client.send(message.encode("utf-8"))
                    print(f"Sent '{message}'")
                    message = ""
            
        else: 
            message += letter
            print(letter, end="")

    try:
        data = client.recv(1)
        print(f"Received: {data}")

        # if data == ACK:
        #     msg = random.choice(messages)
        #     print(f"Sending: {msg}")
        #     client.send(msg)
        #     client.send(b'\r')

        if data == OK:
            print("Recieved OK, taking image")
            img = getImage(imageUrl)
            cv.imwrite(outPath + 'test.png', img)
            print("",end="")

    except:
        print("",end="")