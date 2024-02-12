import RPi.GPIO as GPIO
import time
from scipy.io import wavfile


class CubertAudioPlayer():

    def __init__(self, pwm_pin):

        # setup GPIO
        GPIO.setmode(GPIO.BCM)

        # save pin
        self._pwm_pin = pwm_pin

        # setup pin
        GPIO.setup(pwm_pin, GPIO.OUT)
        
        self._pwm = GPIO.PWM(pwm_pin, 50)

    def __del__(self):
        GPIO.cleanup()

    def play(self):
        self._pwm.start(5)
        time.sleep(1)
        self._pwm.stop()

    def chirp(self):
        self._pwm.ChangeFrequency(20)
        self._pwm.start(5)
        for freq in range(20, 80):
            self._pwm.ChangeFrequency(freq)
            time.sleep(0.01)
        self._pwm.stop()


if __name__ == '__main__':
    audio_pin = 12

    audio = CubertAudioPlayer(audio_pin)

    audio.play()

    time.sleep(1)

    audio.chirp()

    del audio