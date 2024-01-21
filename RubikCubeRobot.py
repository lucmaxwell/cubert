import time

import RPi.GPIO as GPIO

from CubertArm import CubertArm, ArmDirection
from RubikCube import RubikCube, Face
from Button import CubertButton
from Motor import CubertMotor

from enum import Enum


class RobotAction(Enum):
    SPIN_BASE_CLOCKWISE = 0
    SPIN_BASE_COUNTERCLOCKWISE = 1
    GRIP_AND_FLIP = 2
    SPIN_CUBE_CLOCKWISE = 3
    SPIN_CUBE_COUNTERCLOCKWISE = 4


class ArmPosition(Enum):
    BOTTOM = 0
    MIDDLE = 1
    TOP = 2
    DROP_OFF = 3





class RubiksCubeRobot:
    # Constants
    # DO NOT MESS WITH THESE VALUES. YOU WILL BREAK SOMETHING
    MAX_SPEED = 3.3
    MIN_SPEED = 0.000001

    def __init__(self, motors_en_pin, base_motor_pin_list, left_motor_pin_list, right_motor_pin_list, endstop_arm_upper_pin, endstop_arm_lower_pin, manual_button_pin_list, command_button_pin):
        # Initialize
        self.base_motor = CubertMotor(motors_en_pin, base_motor_pin_list)
        self.arm = CubertArm(motors_en_pin, left_motor_pin_list, right_motor_pin_list, endstop_arm_upper_pin, endstop_arm_lower_pin)
        self.manual_buttons = [CubertButton(pin, GPIO.LOW) for pin in manual_button_pin_list]
        self.command_button = CubertButton(command_button_pin, GPIO.LOW)

        # Initialize the cube's state
        self.cube = RubikCube(3)
        self.action_instruction_list = []

    def ai_solve(self):
        # # Create the environment
        # env = RubiksCubeEnv(num_scramble=num_scramble)
        #
        # # Scramble the cube
        # original_obs = env.scramble(num_scramble)
        #
        # # Solve the puzzle
        # # Allow multiple attempts
        # done = False
        # count = 0
        # while count < 3 and not done:
        #     count += 1
        #
        #     # Solve the cube
        #     obs = env.set_observation(original_obs)
        #     while not done:
        #         # Determine action and take step
        #         action, _ = model.predict(obs, deterministic=True)
        #         obs, _, done, _, _ = env.step(action)
        #
        #     # Check if the cube has been solved
        #     done = env.is_solved()
        #
        #     if not multiple_attempts:
        #         break
        #
        # # Return
        # return done
        pass

    def fake_ai_solve(self):
        pass

    def solve(self, cube_state_data):
        # Process the image to get the cube state
        cube_state_obs = self.processing_image(cube_state_data)

        # Initialize self.cube with the cube state
        self.cube.set_state_from_observation(cube_state_obs)

        # Reorient the cube to be the default? Is reorientation needed?
        # Save the spin instructions

        # Solve the cube and get the instructions
        solve_instruction_list = self.solve_cube()

        # Compute the instructions for the robot
        robot_instruction_list = self.get_robot_intructions(solve_instruction_list)

        # Validate that the instructions will solve the cube
        # Optimize the instruction, if the instructions undo each other, remove them

        # Return the instructions for the robot
        pass

    def processing_image(self, new_state):
        return None

    def solve_cube(self):
        return []

    def get_robot_intructions(self, solve_instruction_list):
        return []

    # def spin_base(self, action_direction, correction_enabled=False):
    #     step_delay = self.get_delay(self.spin_speed)
    #     degrees_to_rotate = 90
    #
    #     if self.arm_location != 'TOP' and self.arm_location != 'MIDDLE' and self.gripper_functional:
    #         self.open_hand()
    #         self.move_arm_to('MIDDLE')
    #
    #     GPIO.output(self.motors_base_dir_pin, GPIO.HIGH if my_direction == 'cw' else GPIO.LOW)
    #     steps = int(19200 * degrees_to_rotate / 360)
    #     for _ in range(steps):
    #         GPIO.output(self.motors_base_step_pin, GPIO.HIGH)
    #         time.sleep(step_delay / 1000000.0)  # Convert microseconds to seconds
    #         GPIO.output(self.motors_base_step_pin, GPIO.LOW)
    #         time.sleep(step_delay / 1000000.0)
    #
    #     time.sleep(self.inter_action_delay)
    #
    #     if correction_enabled:
    #         my_direction = 'ccw' if my_direction == 'cw' else 'cw'
    #         GPIO.output(self.motors_base_dir_pin, GPIO.HIGH if my_direction == 'cw' else GPIO.LOW)
    #         self.open_hand()
    #
    #         steps = int(19200 * self.cube_rotation_error / 360.0)
    #         for _ in range(steps):
    #             GPIO.output(self.motors_base_step_pin, GPIO.HIGH)
    #             time.sleep(step_delay / 1000000.0)
    #             GPIO.output(self.motors_base_step_pin, GPIO.LOW)
    #             time.sleep(step_delay / 1000000.0)
    #
    #         time.sleep(self.inter_action_delay)

    def move_arm(self, arm_direction):
        self.arm.move(arm_direction)

    def action(self, action):
        # if action == RobotAction.SPIN_BASE_CLOCKWISE:
        #     self.spin_base(clockwise=True)
        # elif action == RobotAction.SPIN_BASE_COUNTERCLOCKWISE:
        #     self.spin_base(clockwise=False)
        # elif action == RobotAction.GRIP_AND_FLIP:
        #     self.grip_and_flip()
        # elif action == RobotAction.SPIN_CUBE_CLOCKWISE:
        #     self.spin_cube(clockwise=True)
        # elif action == RobotAction.SPIN_CUBE_COUNTERCLOCKWISE:
        #     self.spin_cube(clockwise=False)
        pass


    def doStuffs(self):
        # Get the button pressed
        if self.button.pressed():
            hold_time = 0
            pressed_count = 0

            # Count the number of pressed or hold time
            last_button_state = True
            BUFFER_TIME = 0.5
            reading_start_time = time.time()
            reading_time = time.time()
            hold_time_start = time.time()
            while reading_time - reading_start_time < BUFFER_TIME:
                # Button pressed
                if self.button.pressed():
                    last_button_state = True

                    # Reset the countdown
                    reading_start_time = time.time()

                    # Register the hold time
                    hold_time = reading_time - hold_time_start

                # Button released
                else:
                    if last_button_state:
                        pressed_count += 1
                        last_button_state = False

                        # Start the countdown
                        reading_start_time = time.time()

                # Update the reading time
                reading_time = time.time()

            # Solve
            if pressed_count == 1 and hold_time < 1:
                print("1 pressed")
            # Remember scrambled
            elif pressed_count == 2 and hold_time < 1:
                print("2 pressed")
            # Solve to remember scrambled
            elif pressed_count == 3 and hold_time < 1:
                print("3 pressed")
            # Turn off base light
            elif pressed_count == 1 and 1 <= hold_time < 2:
                print("1 pressed 1 hold")
            # Play victory song
            elif pressed_count == 1 and hold_time >= 2:
                print("1 pressed 2 hold")

    def test(self):
        if self.manual_buttons[0].pressed():
            self.arm.move(ArmDirection.UP)
        if self.manual_buttons[1].pressed():
            self.arm.move(ArmDirection.DOWN)


if __name__ == '__main__':
    # Motor and endstop/current sense pin assignments
    motors_en_pin = 5  # GPIO number for motor enable pin
    motors_base_step_pin = 2  # GPIO number for base step pin
    motors_base_dir_pin = 15  # GPIO number for base direction pin
    motors_arm_left_dir_pin = 0  # GPIO number for arm left direction pin
    motors_arm_left_step_pin = 4  # GPIO number for arm left step pin
    motors_arm_right_dir_pin = 16  # GPIO number for arm right direction pin
    motors_arm_right_step_pin = 17  # GPIO number for arm right step pin

    endstop_hand_openLimit_pin = 18  # GPIO number for arm open limit endstop

    endstop_arm_upperLimit_pin = 23  # GPIO number for arm upper limit endstop
    endstop_arm_lowerLimit_pin = 19  # GPIO number for arm lower limit endstop

    # Manual button pins
    raiseArmButton = 33  # GPIO number for raise arm button
    lowerArmButton = 26  # GPIO number for lower arm button
    openHandButton = 25  # GPIO number for open hand button
    closeHandButton = 32  # GPIO number for close hand button
    spinBaseButton = 27  # GPIO number for spin base button

    # Pin list
    base_motor_pin_list = [motors_base_step_pin, motors_base_dir_pin]
    left_motor_pin_list = [motors_arm_left_step_pin, motors_arm_left_dir_pin]
    right_motor_pin_list = [motors_arm_right_step_pin, motors_arm_right_dir_pin]
    manual_button_pin_list = [raiseArmButton, lowerArmButton, openHandButton, closeHandButton, spinBaseButton]

    # GPIO pins
    BUTTON_PIN = 17

    # GPIO mode
    GPIO.setmode(GPIO.BCM)

    # Setup GPIO pins
    GPIO.setup(motors_en_pin, GPIO.OUT)
    GPIO.setup(motors_base_step_pin, GPIO.OUT)
    GPIO.setup(motors_base_dir_pin, GPIO.OUT)
    GPIO.setup(motors_arm_left_dir_pin, GPIO.OUT)
    GPIO.setup(motors_arm_left_step_pin, GPIO.OUT)
    GPIO.setup(motors_arm_right_dir_pin, GPIO.OUT)
    GPIO.setup(motors_arm_right_step_pin, GPIO.OUT)
    GPIO.setup(endstop_hand_openLimit_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(endstop_arm_upperLimit_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(endstop_arm_lowerLimit_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Setup GPIO manual control button pins
    GPIO.setup(raiseArmButton, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(lowerArmButton, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(openHandButton, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(closeHandButton, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(spinBaseButton, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Setup GPIO command button
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Run
    try:
        print("Running cubert...")
        robot = RubiksCubeRobot(motor_pin_list, endstop_pin_list, manual_button_pin_list, BUTTON_PIN)
        while True:
            # robot.doStuffs()
            robot.test()
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
