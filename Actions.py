import Motor
import time

class CubertActions:

    def __init__(self, motor:Motor.CubertMotor, calibrate_distance=False):
        self.motor = motor

        motor.enable()

        self.motor.home(calibrate_distance)

    def flip(self):
        self.motor.moveGripperToPos(Motor.GripperPosition.BOTTOM)
        self.motor.closeHand()
        self.motor.moveGripperToPos(Motor.GripperPosition.TOP)
        self.motor.openHand()

    def rotateFace(self, rotation:Motor.BaseRotation, direction:Motor.Direction):
        self.motor.moveGripperToPos(Motor.GripperPosition.MIDDLE)
        self.motor.closeHand()
        self.motor.spinBase(rotation, direction, degrees_to_correct=15)
        self.motor.openHand()

    def rotateCube(self, rotation:Motor.BaseRotation, direction:Motor.Direction):
        self.motor.spinBase(rotation, direction)

    

if __name__ == '__main__':
    motor_en_pin = 26
    motor_step_pin = [27, 6, 19]
    motor_dir_pin = [17, 5, 13]

    # End stop for arm
    end_stop_hand_open_pin      = 16  # GPIO number for arm open limit end stop
    end_stop_arm_upperLimit_pin = 20  # GPIO number for arm upper limit end stop
    end_stop_arm_lowerLimit_pin = 21  # GPIO number for arm lower limit end stop

    # initialize motor
    motor = Motor.CubertMotor(motor_en_pin, motor_step_pin, motor_dir_pin, end_stop_arm_upperLimit_pin, end_stop_arm_lowerLimit_pin, end_stop_hand_open_pin)

    actions = CubertActions(motor)

    time.sleep(2)

    actions.flip()
    time.sleep(1)
    actions.rotateCube(Motor.BaseRotation.QUARTER, Motor.Direction.CCW)
    time.sleep(1)
    actions.rotateCube(Motor.BaseRotation.HALF, Motor.Direction.CW)
    time.sleep(1)
    actions.rotateFace(Motor.BaseRotation.QUARTER, Motor.Direction.CW)
    time.sleep(1)
    actions.rotateFace(Motor.BaseRotation.HALF, Motor.Direction.CCW)


    del actions
    del motor

