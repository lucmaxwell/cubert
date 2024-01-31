import time

import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2

class Vision:
    def __init__(self):
        self.camera = PiCamera()

        # Camera parameters
        self.camera.resolution = self.camera.MAX_RESOLUTION
        self.camera.framerate = 32
        rawCapture = PiRGBArray(self.camera, size=(640, 480))

        # Allow the camera to startup
        time.sleep(0.1)

    def capture(self):
        self.camera.capture("./vision/image.jpg")

    def check_red_stripe(image, aoi):
        # Crop to the Area of Interest (AOI)
        x, y, w, h = aoi
        cropped_image = image[y:y + h, x:x + w]

        # Convert to HSV color space
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create a mask for red color
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
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


if __name__ == '__main__':
    vision = CubertVision()

    # Capture image
    vision.camera.capture(vision.rawCapture, format="bgr")
    image = vision.rawCapture.array

    # Check for red stripe
    result = vision.check_red_stripe(image)
    print("Red stripe in the middle:", result)
    