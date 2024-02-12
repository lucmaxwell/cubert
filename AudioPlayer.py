import RPi.GPIO as GPIO



class CubertAudioPlayer():

    def __init__(self, pwm_pin):

        # setup GPIO
        GPIO.setmode(GPIO.BCM)

        # save pin
        self._pwm_pin = pwm_pin

        # setup pin
        GPIO.PWM(pwm_pin, 20)

    def __del__(self):
        GPIO.cleanup()


if __name__ == '__main__':
    audio_pin = 12

    audio = CubertAudioPlayer(audio_pin)