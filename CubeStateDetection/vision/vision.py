import cv2 as cv
import numpy as np
import os

def writeImages(colours):
    cv.imwrite(outPath + 'red.jpg', colours[0])
    cv.imwrite(outPath + 'blue.jpg', colours[1])
    cv.imwrite(outPath + 'green.jpg', colours[2])
    cv.imwrite(outPath + 'orange.jpg', colours[3])
    cv.imwrite(outPath + 'yellow.jpg', colours[4])
    cv.imwrite(outPath + 'white.jpg', colours[5])
    return

names = {
    0: "Red",
    1: "Blue",
    2: "Green",
    3: "Orange",
    4: "Yellow",
    5: "White"
}

BGRs = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 100, 255),
    4: (0, 255, 255),
    5: (255, 255, 255)
}

# Paths
basePath = "./CubeStateDetection/vision/"
imagesPath = basePath + "images/"
outPath = basePath + "output/"

# Create output folder
isExist = os.path.exists(outPath)
if not isExist:
   os.makedirs(outPath)

# Parameters
cube = imagesPath + '/3x3 edge (4).jpg'
edgeLength = 3

# Read in the cube
img = cv.imread(cube)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
height = img.shape[0]
width = img.shape[1]

colours = [0, 0, 0, 0, 0, 0]
red1 = cv.inRange(hsv, (0, 100, 110), (5, 255, 255))
red2 = cv.inRange(hsv, (170, 100, 110), (180, 255, 255))
red = cv.bitwise_or(red1, red2)
colours[0] = red

blue = cv.inRange(hsv, (100, 155, 100), (120, 255, 255))
colours[1] = blue

green = cv.inRange(hsv, (62, 160, 0), (90, 255, 255))
colours[2] = green

orange = cv.inRange(hsv, (8, 160, 100), (13, 255, 255))
colours[3] = orange

yellow = cv.inRange(hsv, (15, 160, 125), (35, 255, 255))
colours[4] = yellow

white = cv.inRange(hsv, (0, 0, 125), (180, 50, 255))
colours[5] = white

writeImages(colours)

# Solve the cube
solution = np.zeros((edgeLength, edgeLength))
outImage = np.zeros((height, width, 3), dtype='uint8')

for i in range(edgeLength):
    for j in range(edgeLength):

        # Mask off one of the 9 colours on the cube face
        mask = np.zeros((height, width, 1), dtype='uint8')
        mask[(height//edgeLength) * i:(height//edgeLength) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = 255

        # Find the colour that has the most pixels in that area
        max =0
        id = 0
        idNum = 0
        for colour in colours:
            masked = cv.bitwise_and(mask, colour)
            sum = np.sum(masked == 255)
            if(sum > max):
                max = sum
                id = idNum
            idNum += 1

        solution[i][j] = id

        # Output
        cv.imwrite(outPath + 'blank.jpg', mask)
        outImage[(height//edgeLength) * i:(height//edgeLength) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = BGRs[id]

print(solution)
cv.imwrite(outPath + 'output.jpg', outImage)