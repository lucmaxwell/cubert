import RPi.GPIO as GPIO
import time


class Button:
    DEBOUNCE_DELAY = 0.05  # 50ms

    def __init__(self, pin, active_state):
        self.pin = pin
        self.active_state = active_state
        self.state = not self.active_state
        self.last_reading_state = not self.active_state
        self.last_debounce_time = 0

        # Setup the GPIO pin
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN if active_state else GPIO.PUD_UP)

    def read_button(self):
        # Read the state of the button
        reading_state = GPIO.input(self.pin)

        # Check if the button state has changed
        if reading_state != self.last_reading_state:
            self.last_reading_state = reading_state
            self.last_debounce_time = time.time()

        # Check if the button is in a stable state after the debounce delay
        elif (time.time() - self.last_debounce_time) > self.DEBOUNCE_DELAY:
            self.state = reading_state

    def pressed(self):
        # Read the button state
        self.read_button()

        # Check if the button is in the active state
        return self.state == self.active_state

    def pressed_for(self, ms):
        # Check if the button is pressed and has been in that state for the specified time
        return self.pressed() and (time.time() - self.last_debounce_time) > ms / 1000

    def hold_time(self):
        pressed_and_hold_time = 0
        if self.pressed():
            pressed_and_hold_time = time.time() - self.last_debounce_time

        # Return
        return pressed_and_hold_time
