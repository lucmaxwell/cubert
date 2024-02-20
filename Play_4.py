from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import math
import Vision


vision = Vision.CubertVision()
Vision.CubertVision.capture()

img = cv2.imread(r'./image.jpg')
# Image Checking
if img is None:
  print("Error: File not found")
  exit(0)
img = cv2.resize(img, (500,500))
 

# Convert to grayscale
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert image to binary
_, thresh = cv2.threshold(img_gs, 85, 200, cv2.THRESH_BINARY)

# Find contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)


# Get rid of small contours
cnts = [x for x in cnts if cv2.contourArea(x) > 12000]

#Get rid of big contours
cnts = [x for x in cnts if cv2.contourArea(x) < 10000000]

n = True
while n == True:
  numcon = len(cnts)
  if numcon > 2:
      cnts.pop(len(cnts)-1)
  else:
     n = False

# Reference object dimensions
ref_object = cnts[0] 

box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)

dist_in_cm = 0.15 #reference length
pixel_per_cm = dist_in_pixel/dist_in_cm 


_, blank= cv2.threshold(img_gs, 0, 200, cv2.THRESH_BINARY)

# Draw contours
output = blank
c1 = []
c2 = []

for cnt in cnts:#cnts_all:
    #Centroids
    M = cv2.moments(cnt)
  
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    c1.append(cX)
    c2.append(cY)

    cv2.circle(output, (cX, cY), 7, (255, 255, 255), -1)

    box = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    cv2.drawContours(output, [box.astype("int")], -1, (0, 0, 255), 2)
    mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
    mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
    wid = euclidean(tl, tr)/pixel_per_cm
    ht = euclidean(tr, br)/pixel_per_cm
     
a = (c1[0]-c1[1])**2
b = (c2[0]-c2[1])**2
dist = math.sqrt(a+b)
#print(pixel_per_cm)
print(dist/pixel_per_cm)


