from flask import Flask
import socket
import threading


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


# Bluetooth connection with the robot
ROBOT_BLUETOOTH_ADDRESS = "40:22:D8:F0:E6:1A"
ROBOT_BLUETOOTH_PORT = 1
robot_client = None


def bluetooth_setup():
    global robot_client
    robot_client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    try:
        robot_client.connect((ROBOT_BLUETOOTH_ADDRESS, ROBOT_BLUETOOTH_PORT))
        print("Connected to Bluetooth device.")
    except Exception as e:
        print(f"Error connecting to Bluetooth device: {e}")


# Setting up Bluetooth in a separate thread
threading.Thread(target=lambda: bluetooth_setup()).start()


# Listening to the rubik cube state from robot
def listen_for_cube_state():
    global robot_client
    if robot_client:
        try:
            while True:
                pass
                # Processing the cube state!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                '''
                data = bt_client.recv(1024)
                if data:
                    cube_state = data.decode()
                    print(f"Received cube state: {cube_state}")
                    process_cube_state(cube_state)
                '''
        except Exception as e:
            print(f"Error receiving data: {e}")


def process_cube_state(cube_state):
    """
    Process the received cube state and take necessary actions.
    """
    # Implement the logic to process the cube state
    # Send the result instructions back to the robot to execute
    pass


if __name__ == '__main__':
    app.run(debug=True)
