import cv2
import numpy as np

img = cv2.imread(r'C:\Users\noahm\OneDrive\Documents\Capstone_Code\Assets\rotated.jpg')  #load an image of a single battery
img = cv2.resize(img, (500,500))
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to grayscale

#inverted binary threshold: 1 for the battery, 0 for the background
_, thresh = cv2.threshold(img_gs, 50, 500, cv2.THRESH_BINARY)
cv2.imshow("out.png", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


mat = np.argwhere(thresh != 0)

mat[:, [0, 1]] = mat[:, [1, 0]]

mat = np.array(mat).astype(np.float32) 

m, e = cv2.PCACompute(mat, mean = np.array([]))
print(e)

angle = np.arctan2(e[0][1], e[0][0]) 
print(angle)
center = tuple(m[0].astype(int))
endpoint1 = tuple(m[0].astype(int) + (5000*e[0]).astype(int))
endpoint2 = tuple(m[0].astype(int) + (5000*e[1]).astype(int))

red_color = (0, 0, 255)

cv2.circle(img, center, 10, red_color)
cv2.line(img, center, endpoint1, red_color)
cv2.line(img, center, endpoint2, red_color)

cv2.imshow("out.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

display = angle*-1*180/np.pi 
print('Misaligned by:')
print(display ,'degrees')