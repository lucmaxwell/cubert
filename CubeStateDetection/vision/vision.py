import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import colorsys

def writeImages(colours):
    cv.imwrite(outPath + '0.jpg', colours[0])
    cv.imwrite(outPath + '1.jpg', colours[1])
    cv.imwrite(outPath + '2.jpg', colours[2])
    cv.imwrite(outPath + '3.jpg', colours[3])
    cv.imwrite(outPath + '4.jpg', colours[4])
    cv.imwrite(outPath + '5.jpg', colours[5])
    return

BGRs = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 100, 255),
    4: (0, 255, 255),
    5: (255, 255, 255)
}

print(f'Working in {os.getcwd()}')

# Parameters
basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"
imagesPath = basePath + "images/"
outPath = basePath + "output/"
image = "new base 4led diffusion.jpg"
cube = imagesPath + image

edgeLength = 3
edgeHeight = 3

# Create output folder
isExist = os.path.exists(outPath)
if not isExist:
   os.makedirs(outPath)

# Read in the cube
img = cv.imread(cube)
img = img.astype(np.uint8)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

height = img.shape[0]
width = img.shape[1]

# Make lists for clustering
inlineHsv = hsv.reshape(height*width, 3)
inlineHsv = inlineHsv.astype(np.int32)
inlineHsv2 = inlineHsv.astype(np.int32)
inlineRgb = rgb.reshape(height*width, 3) / 255

# HSV cylinderical to cartesian coordinates transformation
hue = inlineHsv[:, 0] * np.pi / 180 * 2 # For some reason OpenCV's hue values only go from 0 to 180 so we need to multiply by 2 to get the range 0 to 360
sat = inlineHsv[:, 1]
val = inlineHsv[:, 2]

sat2 = (pow(1.02, sat) - 1) / 155.97 * 255
sat2 = np.array(sat2)
sat2[sat2 > 5] = 255

inlineHsv[:, 0] = np.sin(hue) * sat
inlineHsv[:, 1] = np.cos(hue) * sat
inlineHsv[:, 2] = val

inlineHsv2[:, 0] = np.sin(hue) * sat2
inlineHsv2[:, 1] = np.cos(hue) * sat2
inlineHsv2[:, 2] = val

inline = inlineHsv

# Perform clustering
kmeans = KMeans(n_clusters=6)
kmeans.fit(inline)
labels = kmeans.predict(inline)
labels = labels.reshape((height, width))
colours_pred = kmeans.cluster_centers_

# Convert the cluster centroids from cartesian to cylinderical coordinates
for i in range(len(colours_pred)):
    val = colours_pred[i][2]
    hue = np.arctan2(colours_pred[i][0], colours_pred[i][1])
    if(hue < 0):
        hue = hue + (2*np.pi)
    sat = colours_pred[i][1]/np.cos(hue)

    hue =  hue / np.pi * 180 / 2
    colours_pred[i] = [hue, sat, val]

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

        # Find the colour that has the most pixels in that area
        max =0
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



# Plot the colours
figure = plt.figure()
axis = figure.add_subplot(1,2,1,projection='3d')
axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)

axis = figure.add_subplot(1,2,2,projection='3d')
# axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)
axis.scatter(inlineHsv2[:, 0], inlineHsv2[:, 1], inlineHsv2[:, 2], c=inlineRgb)
# axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)
plt.show()