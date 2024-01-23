import RPi.GPIO as GPIO
import time


class CubertButton:
    DEBOUNCE_DELAY = 0.05  # 50ms

    def __init__(self, pin, active_state):
        self.pin = pin
        self.active_state = active_state
        self.state = not self.active_state
        self.last_reading_state = not self.active_state
        self.last_debounce_time = 0

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


if __name__ == '__main__':
    # Choose GPIO mode (BCM or BOARD)
    GPIO.setmode(GPIO.BCM)

    # Define the pin number where the button is connected
    BUTTON_PIN = 17  # Use the appropriate pin number based on your setup

    # Set up the button pin with a pull-up resistor
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    try:
        while True:
            # Read the button state
            button_state = GPIO.input(BUTTON_PIN)

            # Check if the button is pressed (the pin reads low if pressed)
            if not button_state:
                print("Button Pressed")
                while not GPIO.input(BUTTON_PIN):
                    # Wait for the button to be released to avoid multiple prints
                    time.sleep(0.1)

            # Small delay to avoid busy looping
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Program stopped")

    finally:
        GPIO.cleanup()  # Clean up GPIO settings
