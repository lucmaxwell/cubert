import Motor
import time

class CubertActions:

    _GRIP_STRENGTH  = 350

    def __init__(self, motor:Motor.CubertMotor, calibrate_distance=False):
        self.motor = motor

        self.motor.home(calibrate_distance)

    def flip(self):
        self.motor.moveGripperToPos(Motor.GripperPosition.BOTTOM)
        self.motor.moveGripper(self._GRIP_STRENGTH, Motor.GripperDirection.CLOSE)
        self.motor.moveGripperToPos(Motor.GripperPosition.TOP)
        self.motor.moveGripper(self._GRIP_STRENGTH, Motor.GripperDirection.OPEN)

if __name__ == '__main__':
    motor_en_pin = 26
    motor_step_pin = [27, 6, 19]
    motor_dir_pin = [17, 5, 13]

    # End stop for arm
    end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
    end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
    end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

    # initialize motor
    motor = CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin)

    actions = CubertActions(motor)

    actions.flip()

