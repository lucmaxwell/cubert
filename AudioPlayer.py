import RPi.GPIO as GPIO
import time


class CubertAudioPlayer():

    def __init__(self, pwm_pin):

        # setup GPIO
        GPIO.setmode(GPIO.BCM)

        # save pin
        self._pwm_pin = pwm_pin

        # setup pin
        GPIO.setup(pwm_pin, GPIO.OUT)
        
        self._pwm = GPIO.PWM(pwm_pin, 20)

    def __del__(self):
        GPIO.cleanup()

    def play(self):
        self._pwm.start(1)
        time.sleep(1)
        self._pwm.stop()


if __name__ == '__main__':
    audio_pin = 12

    audio = CubertAudioPlayer(audio_pin)

    audio.play()

    del audio