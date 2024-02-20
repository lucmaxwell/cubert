from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import math
import Vision
import CurrentSensor
import Motor

# Motor Pins
motor_en_pin = 26
motor_step_pin = [27, 6, 19]
motor_dir_pin = [17, 5, 13]

# End stop for arm
end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

sensor = CurrentSensor.CubertCurrentSensor()

motor = Motor.CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin, sensor)

motor.enable()

motor.home()

vision = Vision.CubertVision()
motor.moveGripperToPos(Motor.GripperPosition.MIDDLE, 50)
motor.closeHand()
vision.capture()

#The Image to anaylze
imgac = cv2.imread(r'./images.jpg')

#Reference Masking
img = cv2.imread(r'./images/mask.png')

# Image Checking
if imgac is None:
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

# Reference object dimensions
ref_object = cnts[0] 

box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)

dist_in_cm = 0.095 #reference length
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

print(((sum(tots)/6)/pixel_per_cm)*100, 'mm')


