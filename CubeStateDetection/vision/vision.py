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

# Paths
basePath = "./CubeStateDetection/vision/"
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
height = img.shape[0]
width = img.shape[1]
inline = img.reshape(height*width, 3)

figure = plt.figure()
axis = figure.add_subplot(projection='3d')

kmeans = KMeans(n_clusters=6)
kmeans.fit(inline)
labels = kmeans.predict(inline)
colours_pred = kmeans.cluster_centers_
colours_pred = colours_pred.astype(np.uint8)
print(colours_pred)

# agglomerate = AgglomerativeClustering(linkage="ward", n_clusters=6)
# agglomerate.fit(inline)
# ag_labels = agglomerate.labels_
# ag_labels = ag_labels.reshape((height, width))
# unique, counts = np.unique(ag_labels, return_counts=True)
# print("Aggregate")
# print(np.asarray((unique, counts)).T)

axis.scatter(inline[:, 0], inline[:, 1], inline[:, 2], c=labels)
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
            # masked = cv.bitwise_and(mask, colours[i])
            # masked = (mask == colours[k]).astype(np.uint8) * 255
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
cv.imwrite(outPath + 'output.jpg', outImage)
cv.imwrite(outPath + 'input.jpg', img)