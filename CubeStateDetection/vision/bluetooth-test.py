import socket
import cv2 as cv
import numpy as np
import os
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

basePath = os.getcwd() + "/CubeStateDetection/vision/"
imagesPath = basePath + "images/"
imageUrl = "http://192.168.4.1/capture"
mask = 'mask2.png'
maskPath = imagesPath + mask

def getAllImages(imageUrl, client, maskPath="", writeConsole=False):
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
        
        waitForAcks(client, expectedAcks, writeConsole)

        if(writeConsole):
            print("Cube rotated, waiting 2 seconds for camera to stabilize")

        if(i != 5):
            time.sleep(2)

    return combinedImage, combinedMask

def send(client, message, writeConsole=True):
    try:
        message = message.encode("utf-8")
    except:
        print("", end='')

    client.send(message)
    if(writeConsole):
        print(f"Sent: {message}")

def solve(imageUrl, maskPath="", loadFromDisk=False, writeImages=False):
    
    imageName = "0testing.png"
    maskName = "0testingMask.png"

    print(f"Starting cube solving sequence")

    # Take images
    print("Taking images")
    if( not loadFromDisk):
        cube, mask = getAllImages(imageUrl, client, maskPath, True)
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
    cubeState, outImage = vision.getCubeState(cube, mask, 3, 18, True)
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

def captureCube(imageUrl, client, cubesFolder):
    cube, mask = getAllImages(imageUrl, client)
    num = 0
    name = f"cube{num}.png"
    while(os.path.exists(cubesFolder + name)):
        num += 1
        name = f"cube{num}.png"
    cv.imwrite(cubesFolder + name, cube)

def enableLights(imageUrl, client, writeConsole=False):

    expectedAcks = 1

    lights = np.zeros(4)

    # Find the orientation with the highest lightness
    for i in range(4):
        img = vision.getImage(imageUrl)
        average = np.average(img)
        lights[i] = average
        send(client, y, writeConsole)

        waitForAcks(client, expectedAcks, writeConsole)

        if(i < 4):
            time.sleep(0.25)

    # Spin to the lightest side
    spin = lights.argmax()
    if(spin == 1):
        send(client, y, writeConsole)
    elif(spin == 2):
        send(client, y, writeConsole)
        send(client, y, writeConsole)
    elif(spin == 3):
        send(client, y_, writeConsole)

def waitForAcks(client, expectedAcks, writeConsole=False):
    emptySocket(client)
    client.setblocking(True)

    ackCount = 0
    while ackCount < expectedAcks:
        data = client.recv(1)
        if(data == ACK):
            ackCount += 1
            if(writeConsole):
                print(f'ack {ackCount}/{expectedAcks} received')

    client.setblocking(False)

def emptySocket(client, writeConsole=False):
    data = ""
    try:
        while True:
            data += client.recv(1)
    except:
        if(writeConsole):
            print(f"Emptied socket: {data}")


client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
# client.connect(("40:22:D8:F0:E6:1A", 1))    # ESP32test
client.connect(("40:22:D8:EB:2B:3A", 1))    # ESP32test2
client.setblocking(False)
print("Connected to ESP32 through bluetooth")

message = ''

while True:
    command = input('Input command: ')

    emptySocket(client)

    match command:
        case "solve":
            solve(imageUrl, maskPath)

        case "disk":
            solve(imageUrl, maskPath, True, True)

        case "cap":
            captureCube(imageUrl, client, imagesPath + "cubes/")

        case "data":
            num = 1
            while True:
                print(f"{num}: Waiting for scramble")
                send(client, "s", False)
                waitForAcks(client, 1, True)
                print(f"{num}: Cube scrambled")
                time.sleep(2)

                print(f"{num}: enabling lights")
                enableLights(imageUrl, client)
                print(f"{num}: enabled lights")
                time.sleep(2)

                print(f"{num}: capturing cube")
                captureCube(imageUrl, client, imagesPath + "cubes/")
                print(f"{num}: cube captured")
                time.sleep(1)

                num += 1
                
        case "l":
            enableLights(imageUrl, client)

        case _:
            if command != "":
                send(client, command)

    print(f"\n{command}: Command completed, ready for next command")
