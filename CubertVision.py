from picamera import PiCamera

class CubertVision:
    def __init__(self):
        self.camera = PiCamera()

        # Camera parameters
        self.camera.resolution = self.camera.MAX_RESOLUTION

    def capture(self):
        self.camera.capture("./vision/image.jpg")
