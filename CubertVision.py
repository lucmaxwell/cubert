from picamera import PiCamera

class CubertVision:
    def __init__(self):
        self.camera = PiCamera()

    def capture(self):
        self.camera.capture("./vision/image.jpg")
