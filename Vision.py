import time
from picamera import PiCamera
from picamera.array import PiRGBArray
from cv2 import imwrite, imread, resize, cvtColor, COLOR_BGR2HSV, COLOR_HSV2BGR, COLOR_BGR2RGB
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import urllib.request
import scipy.stats as stats
import plotext as plt


class CubertVision:

    resolution = (64, 48)
    lowerResolution = (45, 45)
    mask = np.full(tuple(list(lowerResolution) + [3]), 1, dtype=np.uint8)

    imagesFolder = "./images/"
    maskName = "mask.png"

    def __init__(self):
        self.camera = PiCamera()

        # Camera parameters
        self.camera.resolution = self.resolution

        # Allow the camera to startup
        time.sleep(2)

        self.mask = self.loadMask(self.imagesFolder + self.maskName)

    def capture(self):
        self.camera.capture("./image.jpg")

    def writeImages(self, colours, outPath):
        for i in range(colours.shape[0]):
            imwrite(outPath + f'{i}.png', colours[i])
        return

    def writeImage(self, fileName, image):
        imwrite(self.imagesFolder + fileName, image)

    def getImage(self):
        size = tuple(list(self.resolution) + [3])

        image = np.empty(size, dtype=np.uint8)
        self.camera.capture(image, 'rgb')
        self.writeImage("0 original.png", image)

        image = resize(image, self.lowerResolution)
        self.writeImage("1 resized.png", image)


        return image

    def hsvToXyz(self, hue, saturation, value):
        x = np.zeros(hue.size)
        y = np.zeros(hue.size)
        z = np.zeros(hue.size)

        x = np.sin(hue) * saturation
        y = np.cos(hue) * saturation
        z = value
        
        return np.array([x, y, z]).T

    def xyzToHsv(self, x, y, z):
        val = z
        hue = np.arctan2(x, y)
        
        hue[hue < 0] = hue[hue < 0] + (2*np.pi)
        sat = y/np.cos(hue)

        hue =  hue / np.pi * 180 / 2
        
        return np.array([hue, sat, val]).T

    def getBlankMask(self, height, width=-1):
        if(width == -1):
            width = height
        return np.full((height, width, 3), 1, dtype=np.uint8)

    def loadMask(self, maskPath):
        imageMask = imread(maskPath)
        imageMask = imageMask.astype(np.uint8)
        imageMask[imageMask != 255] = 0
        imageMask[imageMask == 255] = 1
        imageMask = resize(imageMask, self.lowerResolution)
        self.mask = imageMask

    def getAutoMask(self, rgbCube, min, max):
        hsvCube = cvtColor(rgbCube, COLOR_BGR2HSV)
        autoMask = np.full(hsvCube.shape, 1, dtype=np.uint8)

        autoMask[hsvCube[:, :, 2] <= min] = np.array([0, 0, 0])
        autoMask[hsvCube[:, :, 2] >= max] = np.array([0, 0, 0])

        return autoMask

    def loadCube(self, imagePath):
        return imread(imagePath)
        
    def crop(self, image, y, x, height, width=-1):
        if(width == -1):
            width = height
        
        return image[y:y+height, x:x+width, :]

    def getCubeState(self, rgbCube, mask, cubeletsVertical, cubeletsHorizontal, writeOutput=False, useOriginalAlgorithm=False, facesInImage=6): 
        # Parameters
        clearOutputDirectory = False
        edgeLength = cubeletsHorizontal
        edgeHeight = cubeletsVertical
        numColours = 6

        useCentreCorrection = True

        # Kind of also parameters but not really
        basePath = os.getcwd() + "/CubeStateDetection/vision/"
        outPath = basePath + "output/"

        # Create/clean output folder
        if not os.path.exists(outPath):
            os.makedirs(outPath)

        if clearOutputDirectory:
            files = glob.glob(outPath + '*')
            for f in files:
                os.remove(f)

        rgbCube = rgbCube.astype(np.uint8)
        hsv = cvtColor(rgbCube, COLOR_BGR2HSV)
        rgb = cvtColor(rgbCube, COLOR_BGR2RGB)

        height = rgbCube.shape[0]
        width = rgbCube.shape[1]

        imageMask = mask

        inlineMask = imageMask.reshape(height*width, 3) != 0
        maskedPixels = (imageMask[:, :, 0] == 0)

        # Make lists for clustering
        inlineHsv = hsv.reshape(height*width, 3).astype(np.int32)
        # inlineRgb = rgb.reshape(height*width, 3) / 255

        # HSV cylinderical to cartesian coordinates transformation
        hue = np.copy(inlineHsv[:, 0]) * np.pi / 180 * 2 # For some reason OpenCV's hue values only go from 0 to 180 so we need to multiply by 2 to get the range 0 to 360
        sat = np.copy(inlineHsv[:, 1])
        val = np.copy(inlineHsv[:, 2])
        inlineHsv = self.hsvToXyz(hue, sat, val)

        # Apply the mask to the flattened image before fitting kmeans
        kmeans = KMeans(n_clusters=numColours, n_init=10)
        inlineMasked = inlineHsv[inlineMask]
        inlineMasked = inlineMasked.reshape((inlineMasked.size//3, 3))
        kmeans.fit(inlineMasked)

        # Predict the colour class for each pixel. Masked pixels are predicted but are masked later
        labels = kmeans.predict(inlineHsv)
        labels = labels.reshape((height, width))

        colours_pred = kmeans.cluster_centers_
        colours_pred = self.xyzToHsv(colours_pred[: ,0], colours_pred[:, 1], colours_pred[:, 2])

        # Print debugging images
        colours = np.zeros((numColours, labels.shape[0], labels.shape[1]))

        for i in range(0, numColours):
            colours[i] = (labels == i).astype(np.uint8) * 255
            colours[i][maskedPixels] = 0

        if writeOutput:
            self.writeImages(colours, outPath)

        #############################################
        # Vision V1
        #############################################

        if useOriginalAlgorithm:
            # Solve the cube
            solution = np.zeros((edgeHeight, edgeLength), dtype=np.uint8)
            outImage = np.zeros((edgeHeight, edgeLength, 3), dtype='uint8')
            regionsImage = np.zeros((height, width, 3), dtype='uint8')

            # Centre correction requires some tracking
            if useCentreCorrection:
                centreCounts = np.zeros((6, 6), dtype=np.int32)
                centreIndicies = np.array([1, 4, 7, 10, 13, 16])

            for i in range(edgeHeight):
                for j in range(edgeLength):

                    # Mask off one of the squares on the cube face
                    mask = np.zeros((height, width), dtype='uint8')
                    mask[(height//edgeHeight) * i:(height//edgeHeight) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = 255

                    # Mask off pixels from the pixels from the image mask
                    mask[maskedPixels] = 0

                    # Find the colour that has the most pixels in that area
                    id = stats.mode(labels[mask == 255]).mode

                    # Track the centre stats for centre correction
                    if(i == 1 and j in centreIndicies and useCentreCorrection):
                        index = (np.where(centreIndicies == j))[0][0]
                        for k in range(6):
                            centreCounts[index, k] = (labels[mask == 255] == k).sum()

                    solution[i][j] = id
                    # print(f"({i}, {j}): {id}")
                    
                    # Output
                    regionsImage[mask != 0] = [255/edgeHeight * i, 255/edgeLength * j, 255]
                    outImage[i, j] = colours_pred[id]

            # Apply the centre correction
            if(useCentreCorrection):
                results = np.full(6, -1, dtype=np.int16)
                for i in range(6):
                    winner = results[0] # There is no do-while loop in python so instead do this to make the while condition always fail

                    while winner in results:
                        highest = np.unravel_index(centreCounts.argmax(), centreCounts.shape)
                        centreCounts[highest] = -1
                        centreNum = highest[0]
                        winner = highest[1]

                    results[i] = winner
                    centreCounts[highest[0]] = [-1, -1, -1, -1, -1, -1]
                    solution[1][1 + 3*centreNum] = winner
                    outImage[1, 1 + 3*centreNum] = colours_pred[winner]

        #############################################
        # Vision V2
        #############################################

        else:
            # Solve the cube
            solution = np.zeros((edgeHeight, edgeLength), dtype=np.uint8)
            # outImage = np.zeros((edgeHeight, edgeLength, 3), dtype='uint8')

            outOffset = hsv.shape[0]
            outImage = np.zeros((hsv.shape[0] * 2, hsv.shape[1], hsv.shape[2]), dtype='uint8')
            outImage[0:outOffset, 0:] = hsv

            regionsImage = np.zeros((height, width, 3), dtype='uint8')

            # Get the pixel count of all numColours colours for each cubelet
            pixelCounts = np.zeros((edgeHeight, edgeLength, numColours), dtype=np.int32)
            for i in range(edgeHeight):
                for j in range(edgeLength):

                    # Mask off one of the squares on the cube face
                    mask = np.zeros((height, width), dtype='uint8')
                    mask[(height//edgeHeight) * i:(height//edgeHeight) * (i+1), (width//edgeLength)*j:(width//edgeLength)*(j+1)] = 255

                    # Mask off pixels from the pixels from the image mask
                    mask[maskedPixels] = 0

                    # Track the centre stats for centre correction
                    for k in range(numColours):
                        pixelCounts[i, j, k] = (labels[mask == 255] == k).sum()
            
            # Total number of centres, corners, and edges on each face of the cube
            maxCentres = 1
            maxCorners = 4
            maxEdges = edgeHeight * (edgeLength // facesInImage) - maxCentres - maxCorners

            # These arrays contain the number of cubelets left to be assigned
            # When they reach 0 there are no more cubelets of that colour in that position remaining
            centreCounts = np.zeros(numColours,dtype=np.int32)
            edgeCounts = np.zeros(numColours, dtype=np.int32)
            cornerCounts = np.zeros(numColours, dtype=np.int32)

            centres = np.zeros((edgeHeight, edgeLength), dtype=np.uint32)
            edges = np.zeros((edgeHeight, edgeLength), dtype=np.uint32)
            corners = np.zeros((edgeHeight, edgeLength), dtype=np.uint32)

            # Centres and corners are defined by having special positions 
            centre = np.floor(edgeHeight/2)
            corner = [0, edgeHeight-1]

            # Label all the cubelets centre, corner, or edge
            for i in range(edgeHeight):
                for j in range(edgeLength):
                    if(i == centre and (j % edgeHeight) == centre):
                        # print(f"Centre i:{i}, j:{j}")
                        centres[i, j] = 1
                    elif(i in corner and (j % edgeHeight) in corner):
                        corners[i, j] = 1  
                        # print(f"Corner i:{i}, j:{j}")
                    else:
                        edges[i, j] = 1
                        # print(f"Edge i:{i}, j:{j}")
                    
            # Solve the cube
            for i in range(edgeHeight):
                for j in range(edgeLength):
                    colourCount = 0
                    maxCount = -1

                    while colourCount > maxCount:
                        # Find the cubelet that is most confident in its colour
                        highest = np.unravel_index(pixelCounts.argmax(), pixelCounts.shape)
                        
                        y = highest[0]
                        x = highest[1]
                        colour = highest[2]

                        # Determine if this cubelet is a centre, corner, or edge 
                        # (determine its position)
                        if(centres[y, x] == 1):
                            position = centres
                            maxCount = maxCentres
                            counts = centreCounts    
                            # print(f"Centre y:{y}, x:{x}")
                        elif(corners[y, x] == 1):
                            position = corners
                            maxCount = maxCorners
                            counts = cornerCounts    
                            # print(f"Corner y:{y}, x:{x}")
                        else:
                            position = edges
                            maxCount = maxEdges
                            counts = edgeCounts
                            # print(f"Edge y:{y}, x:{x}")

                        counts[colour] += 1
                        colourCount = counts[colour]

                        # Remove all of this colour at this position if there's been too many
                        if(counts[colour] >= maxCount):
                            pixelCounts[position == 1, colour] = np.full(maxCount * facesInImage, -1)
                    
                    # Remove this cubelet's pixel counts, save this cubelet's colour
                    pixelCounts[y, x] = np.full(numColours, -1)
                    solution[y, x] = colour
                    # outImage[y, x] = colours_pred[colour]                
                    outImage[outOffset + y*(hsv.shape[0] // cubeletsVertical):outOffset + (y+1)*(hsv.shape[0] // cubeletsVertical), x * (hsv.shape[1] // cubeletsHorizontal): (x+1) * (hsv.shape[1] // cubeletsHorizontal)] = colours_pred[colour]                

        # Write results
        if writeOutput:
            imwrite(outPath + 'output.png', cvtColor(outImage, COLOR_HSV2BGR))
            imwrite(outPath + 'input.png', rgbCube)
            imwrite(outPath + 'mask.png', imageMask * 255)
            imwrite(outPath + 'masked.png', rgbCube * imageMask)
            imwrite(outPath + 'maskedRegions.png', regionsImage)

        outImage = cvtColor(outImage, COLOR_HSV2BGR)
        return solution, outImage

# if __name__ == '__main__':
#     vision = CubertVision()

#     vision.capture()
#     vision.camera.start_preview()
#     plt.image_plot("./image.jpg")
#     plt.show()