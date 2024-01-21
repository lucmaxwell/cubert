import RPi.GPIO as GPIO
import time

# Choose GPIO mode (BCM or BOARD)
GPIO.setmode(GPIO.BOARD)

# Define the pin number where the button is connected
BUTTON_PIN = 17  # Use the appropriate pin number based on your setup

# Set up the button pin with a pull-up resistor
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

try:
    while True:
        # Read the button state
        button_state = GPIO.input(BUTTON_PIN)

        # Check if the button is pressed (the pin reads low if pressed)
        if button_state == False:
            print("Button Pressed")
            while GPIO.input(BUTTON_PIN) == False:
                # Wait for the button to be released to avoid multiple prints
                time.sleep(0.1)

        # Small delay to avoid busy looping
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program stopped")

finally:
    GPIO.cleanup()  # Clean up GPIO settings
