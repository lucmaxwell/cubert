import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

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

# Paths
basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"

# basePath = "./"
imagesPath = basePath + "images/"
outPath = basePath + "output/"
image = "combined.jpg"

# Create output folder
isExist = os.path.exists(outPath)
if not isExist:
   os.makedirs(outPath)

# Parameters
cube = imagesPath + image
edgeLength = 18
edgeHeight = 3

# Read in the cube
img = cv.imread(cube)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
height = img.shape[0]
width = img.shape[1]

# Change to euclidian HSV coordinates
inlineHsv = hsv.reshape(height*width, 3)
inlineHsv = inlineHsv.astype(np.int32)

inlineRgb = rgb.reshape(height*width, 3) / 255

print(inlineHsv[0:10])
hue = inlineHsv[:, 0] * np.pi / 180 * 2
sat = inlineHsv[:, 1] #/ 255 * 100
val = inlineHsv[:, 2] #/ 255 * 100

inlineHsv[:, 0] = np.sin(hue) * sat
inlineHsv[:, 1] = np.cos(hue) * sat
inlineHsv[:, 2] = val

# inline = img.reshape(height*width, 3)
inline = inlineHsv

figure = plt.figure()
axis = figure.add_subplot(projection='3d')

kmeans = KMeans(n_clusters=6)
kmeans.fit(inline)
labels = kmeans.predict(inline)
colours_pred = kmeans.cluster_centers_
colours_pred = colours_pred.astype(np.uint8)
print(colours_pred)

axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=inlineRgb)
# plt.show()

labels = labels.reshape((height, width))
unique, counts = np.unique(labels, return_counts=True)
print("KMeans")
print(np.asarray((unique, counts)).T)

colours = [0, 0, 0, 0, 0, 0]
for i in range(0, 6):
    colours[i] = (labels == i).astype(np.uint8) * 255
    # colours[i] = (ag_labels == i).astype(np.uint8) * 255
    # print(colours[i])

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
        print(f"{i}, {j}: {id}")

        # Output
        cv.imwrite(outPath + 'blank.jpg', mask)
        outImage[(height//edgeHeight) * i:(height//edgeHeight) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = colours_pred[id]

print(solution)
cv.imwrite(outPath + 'output.jpg', cv.cvtColor(outImage, cv.COLOR_HSV2BGR_FULL) )
cv.imwrite(outPath + 'input.jpg', img)