import cv2 as cv
import numpy as np
import os
import urllib.request

def getImage(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(arr, -1) # 'Load it as it is'
    return img

print(f'Working in {os.getcwd()}')

# Parameters
basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"
imagesPath = basePath + "images/"
imageUrl = "http://192.168.4.1/capture"

# Create output folder
isExist = os.path.exists(imagesPath)
if not isExist:
   os.makedirs(imagesPath)

img = getImage(imageUrl)

top = 160
left = 260
height = 324
width = height

# img = img[top:top+height, left:left+width, :]

number = 0
name = "cob"

image = f"{name}{number}.jpg"

while(os.path.exists(imagesPath + image)):
    number += 1
    image = f"{name}{number}.jpg"

cv.imwrite(imagesPath + image, img)
