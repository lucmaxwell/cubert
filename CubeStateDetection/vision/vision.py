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
useMask = True
image = "cob3.jpg"
mask = 'mask.png'
edgeLength = 3
edgeHeight = 3

basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"
imagesPath = basePath + "images/"
outPath = basePath + "output/"
imageUrl = "http://192.168.4.1/capture"
cube = imagesPath + image
maskPath = imagesPath + mask

# Create/clean output folder
if not os.path.exists(outPath):
   os.makedirs(outPath)

# files = glob.glob(outPath + '*')
# for f in files:
#     os.remove(f)

if True:
    # Read in the cube
    if(useUrl):
        img = getImage(imageUrl)
        
        top = 162
        left = 249
        height = 324
        width = height

        img = img[top:top+height, left:left+width, :]
        # cv.imshow('asdf', img)
        # if cv.waitKey() & 0xff == 27: quit()
        
        # 239 from left, 113 from top, 377x377
    else:
        img = cv.imread(cube)

    img = img.astype(np.uint8)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height = img.shape[0]
    width = img.shape[1]

    if(useMask):
        imageMask = cv.imread(maskPath)
        imageMask = imageMask.astype(np.uint8)
        imageMask[imageMask != 255] = 0
        imageMask[imageMask == 255] = 1
        inlineMask = imageMask.reshape(height*width, 3)

        # cv.imshow("asdf", img * imageMask)
        # if cv.waitKey() & 0xff == 27: quit()

    # Make lists for clustering
    inlineHsv = hsv.reshape(height*width, 3).astype(np.int32)
    inlineHsv2 = np.copy(inlineHsv)
    inlineRgb = rgb.reshape(height*width, 3) / 255

    # HSV cylinderical to cartesian coordinates transformation
    hue = np.copy(inlineHsv[:, 0]) * np.pi / 180 * 2 # For some reason OpenCV's hue values only go from 0 to 180 so we need to multiply by 2 to get the range 0 to 360
    sat = np.copy(inlineHsv[:, 1])
    val = np.copy(inlineHsv[:, 2])

    # Alternate saturation
    sat2 = (pow(1.02, sat) - 1) / 155.97 * 255
    sat2 = np.array(sat2)
    sat2[sat2 > 5] = 255
    
    inlineHsv = hsvToXyz(hue, sat, val)
    inlineHsv2 = hsvToXyz(hue, sat2, val)
    inline = inlineHsv

    kmeans = KMeans(n_clusters=6, n_init=10)
    if(useMask):
        masked = inline[inlineMask != 0]
        masked = masked.reshape((masked.size//3, 3))
        kmeans.fit(masked)
    else:
        kmeans.fit(inline)

    # kmeans.fit(inline)
    labels = kmeans.predict(inline)
    labels = labels.reshape((height, width))

    colours_pred = kmeans.cluster_centers_
    colours_pred = xyzToHsv(colours_pred[: ,0], colours_pred[:, 1], colours_pred[:, 2])

    # Print debugging images
    colours = [0, 0, 0, 0, 0, 0]
    for i in range(0, 6):
        colours[i] = (labels == i).astype(np.uint8) * 255
    writeImages(colours)

    # Solve the cube
    solution = np.zeros((edgeHeight, edgeLength))
    outImage = np.zeros((height, width, 3), dtype='uint8')

    for i in range(edgeHeight):
        for j in range(edgeLength):

            # Mask off one of the 9 colours on the cube face
            mask = np.zeros((height, width), dtype='uint8')
            mask[(height//edgeHeight) * i:(height//edgeHeight) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = 255

            if(useMask):
                mask[imageMask[:, :, 0] == 0] = 0

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
    if(useMask):
        cv.imwrite(outPath + 'masked.jpg', img * imageMask)


# Plot the colours
figure = plt.figure()
axis = figure.add_subplot(1,2,1,projection='3d')
axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)

axis = figure.add_subplot(1,2,2,projection='3d')
# axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)
axis.scatter(inlineHsv2[:, 0], inlineHsv2[:, 1], inlineHsv2[:, 2], c=inlineRgb)
# axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)
# plt.show()