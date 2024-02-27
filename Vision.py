import time
from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import urllib.request
import scipy.stats as stats
import imutils
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import math

class CubertVision:

    lowerResolution = (45, 45)
    mask = np.full(tuple(list(lowerResolution) + [3]), 1, dtype=np.uint8)

    imagesFolder = "./images/"
    maskName = "mask.png"

    def __init__(self):
        self.camera = Picamera2()
        camera_config = self.camera.create_still_configuration(lores={"size": (640, 480)}, display="lores")
        self.camera.configure(camera_config)
        # self.camera.set_controls({"AnalogueGain": 1.0})
        self.camera.start()
        time.sleep(2)

        self.loadMask(self.imagesFolder + self.maskName)

    def capture(self):
        self.camera.capture_file("./image.jpg")

    def getCubletSize(self):
        #The Image to anaylze
        imgac = cv.imread(r'./image.jpg')

        #Reference Masking
        img = cv.imread(r'./images/mask.png')

        # Image Checking
        if imgac is None:
            print("Error: File not found")
            return -1

        img = cv.resize(img, (500,500))
        

        # Convert to grayscale
        img_gs = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Convert image to binary
        _, thresh = cv.threshold(img_gs, 85, 200, cv.THRESH_BINARY)

        # Find contours
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        (cnts, _) = contours.sort_contours(cnts)

        # Reference object dimensions
        ref_object = cnts[0] 

        box = cv.minAreaRect(ref_object)
        box = cv.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)

        dist_in_cm = 0.095 #reference length
        pixel_per_cm = dist_in_pixel/dist_in_cm 


        _, blank= cv.threshold(img_gs, 0, 200, cv.THRESH_BINARY)

        # Draw contours
        output = blank
        c1 = []
        c2 = []

        for cnt in cnts:#cnts_all:
            #Centroids
            M = cv.moments(cnt)
        
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            c1.append(cX)
            c2.append(cY)

            cv.circle(output, (cX, cY), 7, (255, 255, 255), -1)

            box = cv.minAreaRect(cnt)
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            cv.drawContours(output, [box.astype("int")], -1, (0, 0, 255), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
            wid = euclidean(tl, tr)/pixel_per_cm
            ht = euclidean(tr, br)/pixel_per_cm
            
        a = []
        b = []

        a.append((c1[0]-c1[1])**2)
        b.append( (c2[0]-c2[1])**2)

        a.append ((c1[1]-c1[2])**2)
        b.append((c2[1]-c2[2])**2)

        a.append ((c1[3]-c1[4])**2)
        b.append((c2[3]-c2[4])**2)

        a.append ((c1[4]-c1[5])**2)
        b.append((c2[4]-c2[5])**2)

        a.append((c1[6]-c1[7])**2)
        b.append((c2[6]-c2[7])**2)

        a.append((c1[7]-c1[8])**2)
        b.append((c2[7]-c2[8])**2)

        tots = []
        n = 0
        while n < len(a):
            tots.append(math.sqrt(a[0] + b[0]))
            n = n + 1

        # print(((sum(tots)/6)/pixel_per_cm)*100, 'mm')
        return ((sum(tots)/6)/pixel_per_cm)*100

    def writeImages(self, colours):
        for i in range(colours.shape[0]):
            cv.imwrite(self.imagesFolder + f'{i}.png', colours[i])
        return

    def writeImage(self, fileName, image):
        cv.imwrite(self.imagesFolder + fileName, image)

    def getImage(self):
        # size = (self.resolution[1], self.resolution[0], 3)

        # image = np.empty(size, dtype=np.uint8)
        image = self.camera.capture_array()
        # self.camera.capture(image, 'bgr')
        self.writeImage("0 original.png", image)

        image = cv.resize(image, self.lowerResolution)
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
        imageMask = cv.imread(maskPath)
        imageMask = imageMask.astype(np.uint8)
        imageMask[imageMask != 255] = 0
        imageMask[imageMask == 255] = 1
        imageMask = cv.resize(imageMask, self.lowerResolution)
        self.mask = imageMask

    def getAutoMask(self, rgbCube, min, max):
        hsvCube = cv.cvtColor(rgbCube, cv.COLOR_BGR2HSV)
        autoMask = np.full(hsvCube.shape, 1, dtype=np.uint8)

        autoMask[hsvCube[:, :, 2] <= min] = np.array([0, 0, 0])
        autoMask[hsvCube[:, :, 2] >= max] = np.array([0, 0, 0])

        return autoMask

    def loadCube(self, imagePath):
        return cv.imread(imagePath)
        
    def crop(self, image, y, x, height, width=-1):
        if(width == -1):
            width = height
        
        return image[y:y+height, x:x+width, :]

    def getCubeState(self, rgbCube, mask, cubeletsVertical, cubeletsHorizontal, writeOutput=False, facesInImage=6): 
        # Parameters
        clearOutputDirectory = False
        edgeLength = cubeletsHorizontal
        edgeHeight = cubeletsVertical
        numColours = 6

        # Create/clean output folder
        if not os.path.exists(self.imagesFolder):
            os.makedirs(self.imagesFolder)

        if clearOutputDirectory:
            files = glob.glob(self.imagesFolder + '*')
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
            self.writeImages(colours)

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
            cv.imwrite(self.imagesFolder + 'output.png', cv.cvtColor(outImage, cv.COLOR_HSV2BGR))
            cv.imwrite(self.imagesFolder + 'input.png', rgbCube)
            cv.imwrite(self.imagesFolder + 'usedMask.png', imageMask * 255)
            cv.imwrite(self.imagesFolder + 'masked.png', rgbCube * imageMask)
            cv.imwrite(self.imagesFolder + 'maskedRegions.png', regionsImage)

        outImage = cv.cvtColor(outImage, cv.COLOR_HSV2BGR)
        return solution, outImage