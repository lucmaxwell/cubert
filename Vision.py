import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2 as cv
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import urllib.request
import scipy.stats as stats


class CubertVision:
    def __init__(self):
        self.camera = PiCamera()

        # Camera parameters
        self.camera.resolution = self.camera.MAX_RESOLUTION
        self.camera.framerate = 32
        self.rawCapture = PiRGBArray(self.camera, size=(480, 640))

        # Allow the camera to startup
        time.sleep(0.1)

    def capture(self):
        self.camera.capture("./image.jpg")

    def check_red_stripe(image, aoi):
        # Crop to the Area of Interest (AOI)
        x, y, w, h = aoi
        cropped_image = image[y:y + h, x:x + w]

        # Convert to HSV color space
        hsv = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)

        # Define range for red color
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask for red color
        mask1 = cv.inRange(hsv, lower_red, upper_red)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Check if there is a red stripe in the middle of the AOI
        height, width = cropped_image.shape[:2]
        middle = height // 2
        tolerance = 10  # Adjust tolerance for the thickness of the stripe

        # Extract the middle section of the mask
        middle_section = mask[middle - tolerance:middle + tolerance, :]

        # Check if there's a significant amount of red in the middle section
        if np.sum(middle_section) > (tolerance * width * 255 * 0.5):  # 50% red in the stripe area
            return True
        else:
            return False


    def writeImages(colours, outPath):
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

    def getBlankMask(height, width=-1):
        if(width == -1):
            width = height
        return np.full((height, width, 3), 1, dtype=np.uint8)

    def loadMask(maskPath):
        imageMask = cv.imread(maskPath)
        imageMask = imageMask.astype(np.uint8)
        imageMask[imageMask != 255] = 0
        imageMask[imageMask == 255] = 1
        return imageMask

    def getAutoMask(rgbCube, min, max):
        hsvCube = cv.cvtColor(rgbCube, cv.COLOR_BGR2HSV)
        autoMask = np.full(hsvCube.shape, 1, dtype=np.uint8)

        autoMask[hsvCube[:, :, 2] <= min] = np.array([0, 0, 0])
        autoMask[hsvCube[:, :, 2] >= max] = np.array([0, 0, 0])

        return autoMask

    def loadCube(imagePath):
        return cv.imread(imagePath)
        
    def crop(image, y, x, height, width=-1):
        if(width == -1):
            width = height
        
        return image[y:y+height, x:x+width, :]

    def getCubeState(rgbCube, mask, cubeletsVertical, cubeletsHorizontal, writeOutput=False, useOriginalAlgorithm=False, facesInImage=6): 
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
        hsv = cv.cvtColor(rgbCube, cv.COLOR_BGR2HSV)
        rgb = cv.cvtColor(rgbCube, cv.COLOR_BGR2RGB)

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

        if writeOutput:
            writeImages(colours, outPath)

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
            cv.imwrite(outPath + 'output.png', cv.cvtColor(outImage, cv.COLOR_HSV2BGR))
            cv.imwrite(outPath + 'input.png', rgbCube)
            cv.imwrite(outPath + 'mask.png', imageMask * 255)
            cv.imwrite(outPath + 'masked.png', rgbCube * imageMask)
            cv.imwrite(outPath + 'maskedRegions.png', regionsImage)

        outImage = cv.cvtColor(outImage, cv.COLOR_HSV2BGR)
        return solution, outImage

if __name__ == '__main__':
    vision = CubertVision()

    vision.capture()

    # Capture image
    # vision.camera.capture(vision.rawCapture, format="bgr")
    # image = vision.rawCapture.array

    # Check for red stripe
    # result = vision.check_red_stripe(image)
    # print("Red stripe in the middle:", result)
    