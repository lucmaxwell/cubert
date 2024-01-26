import cv2 as cv
import numpy as np
import os
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

while(os.path.exists(cubeFolder + name)):
    num += 1
    name = f"cube{num}.png"

mask = vision.loadMask(maskPath)

for i in range(num):
    cubePath = cubeFolder + f"cube{i}.png"
    cube = vision.loadCube(cubePath)
    # autoMask = vision.getAutoMask(cube, 75, 512)
    # combinedMask = mask * autoMask
    combinedMask = mask

    solution, outImage = vision.getCubeState(cube, combinedMask, 3, 18, True)
    cv.imwrite(cubeFolder + f"solution{i}.jpg", outImage)
    print(f"{i}: done")