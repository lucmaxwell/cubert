import cv2 as cv
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import urllib.request

def writeImages(colours):
    cv.imwrite(outPath + '0.jpg', colours[0])
    cv.imwrite(outPath + '1.jpg', colours[1])
    cv.imwrite(outPath + '2.jpg', colours[2])
    cv.imwrite(outPath + '3.jpg', colours[3])
    cv.imwrite(outPath + '4.jpg', colours[4])
    cv.imwrite(outPath + '5.jpg', colours[5])
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
image = "bruno1.png"
mask = 'mask.png'
edgeLength = 3
edgeHeight = 3

useMask = False
useMaskRange = True
maskMin = 100
maskMax = 240

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
        top = 162
        left = 249
        height = 324
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
    if(useMaskRange):
        imageMask[hsv[:, :, 2] < maskMin] = np.array([0, 0, 0])
        imageMask[hsv[:, :, 2] > maskMax] = np.array([0, 0, 0])

    inlineMask = imageMask.reshape(height*width, 3)
    maskedPixels = (imageMask[:, :, 0] == 0)

    # Make lists for clustering
    inlineHsv = hsv.reshape(height*width, 3).astype(np.int32)
    inlineRgb = rgb.reshape(height*width, 3) / 255

    # HSV cylinderical to cartesian coordinates transformation
    hue = np.copy(inlineHsv[:, 0]) * np.pi / 180 * 2 # For some reason OpenCV's hue values only go from 0 to 180 so we need to multiply by 2 to get the range 0 to 360
    sat = np.copy(inlineHsv[:, 1])
    val = np.copy(inlineHsv[:, 2])
    inlineHsv = hsvToXyz(hue, sat, val)
    inline = inlineHsv

    # Apply the mask to the flattened image before fitting kmeans
    kmeans = KMeans(n_clusters=6, n_init=10)
    masked = inline[inlineMask != 0]
    masked = masked.reshape((masked.size//3, 3))
    kmeans.fit(masked)

    # Predict the colour class for each pixel. Masked pixels are predicted but are masked later
    labels = kmeans.predict(inline)
    labels = labels.reshape((height, width))

    colours_pred = kmeans.cluster_centers_
    colours_pred = xyzToHsv(colours_pred[: ,0], colours_pred[:, 1], colours_pred[:, 2])

    # Print debugging images
    colours = [0, 0, 0, 0, 0, 0]
    for i in range(0, 6):
        colours[i] = (labels == i).astype(np.uint8) * 255
        colours[i][maskedPixels] = 0
    writeImages(colours)

    # Solve the cube
    solution = np.zeros((edgeHeight, edgeLength))
    outImage = np.zeros((height, width, 3), dtype='uint8')

    for i in range(edgeHeight):
        for j in range(edgeLength):

            # Mask off one of the 9 squares on the cube face
            mask = np.zeros((height, width), dtype='uint8')
            mask[(height//edgeHeight) * i:(height//edgeHeight) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = 255

            # Mask off pixels from the pixels from the image mask
            mask[maskedPixels] = 0

            # Find the colour that has the most pixels in that area
            max = 0
            id = 0
            idNum = 0
            for k in range(len(colours)):
                masked = np.multiply(mask, colours[k])

                sum = np.sum(masked > 0)
                if(sum > max):
                    max = sum
                    id = idNum
                idNum += 1

            solution[i][j] = id
            print(f"({i}, {j}): {id}")

            # Output
            cv.imwrite(outPath + 'blank.jpg', mask)
            outImage[(height//edgeHeight) * i:(height//edgeHeight) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = colours_pred[id]

    # Write results
    print(solution)
    cv.imwrite(outPath + 'output.jpg', cv.cvtColor(outImage, cv.COLOR_HSV2BGR))
    cv.imwrite(outPath + 'input.jpg', img)
    cv.imwrite(outPath + 'mask.jpg', imageMask * 255)
    cv.imwrite(outPath + 'masked.jpg', img * imageMask)

# Plot the colours
# figure = plt.figure()
# axis = figure.add_subplot(1,2,1,projection='3d')
# axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)

# axis = figure.add_subplot(1,2,2,projection='3d')
# axis.scatter(inlineHsv2[:, 0], inlineHsv2[:, 1], inlineHsv2[:, 2], c=inlineRgb)
# plt.show()