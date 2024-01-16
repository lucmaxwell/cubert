import cv2 as cv
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import urllib.request
import scipy.stats as stats

def writeImages(colours):
    for i in range(colours.shape[0]):
        cv.imwrite(outPath + f'{i}.png', colours[i])
    return

def getImage(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(arr, -1) # 'Load it as it is'
    return img

def hsvToXyz(hue, saturation, value):
    x = np.zeros(hue.size)
    y = np.zeros(hue.size)
    z = np.zeros(hue.size)

    x = np.sin(hue) * saturation
    y = np.cos(hue) * saturation
    z = value
    
    return np.array([x, y, z]).T

def xyzToHsv(x, y, z):
    val = z
    hue = np.arctan2(x, y)
    
    hue[hue < 0] = hue[hue < 0] + (2*np.pi)
    sat = y/np.cos(hue)

    hue =  hue / np.pi * 180 / 2
    
    return np.array([hue, sat, val]).T

# Parameters
useUrl = False
clearOutputDirectory = False
image = "test2 modified.jpg"
mask = 'mask.png'
edgeLength = 3
edgeHeight =3
numColours = 6

useMask = False
useAutoMask = False
maskMin = 75
maskMax = 255

# Kind of also parameters but not really
basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"
imagesPath = basePath + "images/"
outPath = basePath + "output/"
imageUrl = "http://192.168.4.1/capture"
cube = imagesPath + image
maskPath = imagesPath + mask

# Create/clean output folder
if not os.path.exists(outPath):
   os.makedirs(outPath)

if clearOutputDirectory:
    files = glob.glob(outPath + '*')
    for f in files:
        os.remove(f)

# Change to while for real time processing
if True:
    # Read in the cube
    if(useUrl):
        img = getImage(imageUrl)
        
        # 239 from left, 113 from top, 324x324
        top = 150
        left = 220
        height = 230
        width = height

        img = img[top:top+height, left:left+width, :]
    else:
        img = cv.imread(cube)

    img = img.astype(np.uint8)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height = img.shape[0]
    width = img.shape[1]

    # Load the mask
    if(useMask):
        imageMask = cv.imread(maskPath)
        imageMask = imageMask.astype(np.uint8)
        imageMask[imageMask != 255] = 0
        imageMask[imageMask == 255] = 1
    else:
        imageMask = np.copy(img)
        imageMask[:, :, :] = np.array([1,1,1])

    # Mask out the darkest and lightest pixels from the image
    if(useAutoMask):
        imageMask[hsv[:, :, 2] < maskMin] = np.array([0, 0, 0])
        imageMask[hsv[:, :, 2] > maskMax] = np.array([0, 0, 0])

    inlineMask = imageMask.reshape(height*width, 3) != 0
    maskedPixels = (imageMask[:, :, 0] == 0)

    # Make lists for clustering
    inlineHsv = hsv.reshape(height*width, 3).astype(np.int32)
    inlineRgb = rgb.reshape(height*width, 3) / 255

    # HSV cylinderical to cartesian coordinates transformation
    hue = np.copy(inlineHsv[:, 0]) * np.pi / 180 * 2 # For some reason OpenCV's hue values only go from 0 to 180 so we need to multiply by 2 to get the range 0 to 360
    sat = np.copy(inlineHsv[:, 1])
    val = np.copy(inlineHsv[:, 2])
    inlineHsv = hsvToXyz(hue, sat, val)

    # Apply the mask to the flattened image before fitting kmeans
    kmeans = KMeans(n_clusters=numColours, n_init=10)
    inlineMasked = inlineHsv[inlineMask]
    inlineMasked = inlineMasked.reshape((inlineMasked.size//3, 3))
    kmeans.fit(inlineMasked)

    # Predict the colour class for each pixel. Masked pixels are predicted but are masked later
    labels = kmeans.predict(inlineHsv)
    labels = labels.reshape((height, width))

    colours_pred = kmeans.cluster_centers_
    colours_pred = xyzToHsv(colours_pred[: ,0], colours_pred[:, 1], colours_pred[:, 2])

    # Print debugging images
    colours = np.zeros((numColours, labels.shape[0], labels.shape[1]))

    for i in range(0, numColours):
        colours[i] = (labels == i).astype(np.uint8) * 255
        colours[i][maskedPixels] = 0
    writeImages(colours)

    # Solve the cube
    solution = np.zeros((edgeHeight, edgeLength))
    outImage = np.zeros((edgeHeight, edgeLength, 3), dtype='uint8')
    regionsImage = np.zeros((height, width, 3), dtype='uint8')

    for i in range(edgeHeight):
        for j in range(edgeLength):

            # Mask off one of the 9 squares on the cube face
            mask = np.zeros((height, width), dtype='uint8')
            mask[(height//edgeHeight) * i:(height//edgeHeight) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = 255

            # Mask off pixels from the pixels from the image mask
            mask[maskedPixels] = 0

            # Find the colour that has the most pixels in that area
            id = stats.mode(labels[mask == 255]).mode

            solution[i][j] = id
            print(f"({i}, {j}): {id}")
            
            # Output
            regionsImage[mask != 0] = [255/edgeHeight * i, 255/edgeLength * j, 255]
            outImage[i, j] = colours_pred[id]

    # Write results
    print(solution)
    cv.imwrite(outPath + 'output.png', cv.cvtColor(outImage, cv.COLOR_HSV2BGR))
    cv.imwrite(outPath + 'input.png', img)
    cv.imwrite(outPath + 'mask.png', imageMask * 255)
    cv.imwrite(outPath + 'masked.png', img * imageMask)
    cv.imwrite(outPath + 'maskedRegions.png', regionsImage)

# Plot the colours
# figure = plt.figure()
# axis = figure.add_subplot(1,1,1,projection='3d')
# axis.scatter(inlineMasked[:, 0], inlineMasked[:, 1], inlineMasked[:, 2], c=inlineRgb[inlineMask].reshape((inlineMasked.size//3, 3)))
# plt.show()