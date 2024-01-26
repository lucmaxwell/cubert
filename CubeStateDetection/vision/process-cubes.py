import cv2 as cv
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
import scipy.stats as stats
import vision

basePath = os.getcwd() + "/CubeStateDetection/vision/"
cubeFolder = basePath + "/images/cubes/"
imagesPath = basePath + "images/"
mask = 'mask3.png'
maskPath = imagesPath + mask

num = 0
name = f"cube{num}.png"

height = 339
if(len(sys.argv) > 1):
    height = int(sys.argv[1])

length = height * 6

while(os.path.exists(cubeFolder + name)):
    num += 1
    name = f"cube{num}.png"

mask = vision.loadMask(maskPath)
mask = cv.resize(mask, (length, height))

for i in range(num):
    cubePath = cubeFolder + f"cube{i}.png"
    cube = vision.loadCube(cubePath)
    cube = cv.resize(cube, (length, height))
    # autoMask = vision.getAutoMask(cube, 75, 512)
    # combinedMask = mask * autoMask
    combinedMask = mask

    solution, outImage = vision.getCubeState(cube, combinedMask, 3, 18, True)
    cv.imwrite(cubeFolder + f"solution{i}.jpg", outImage)
    print(f"{i}: done")