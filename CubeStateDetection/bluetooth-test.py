import socket
import time

y = 0
y_ = 1
b = 10
b_ = 10
x_ = 20
req = 69
ack = 999

test = [y, y_, b, b_, x_]

client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
client.connect(("40:22:D8:F0:E6:1A", 1))

print("Connected?")

while True:
    data = client.recv(9)
    if not data:
        print(f"No message received, doing nothing")
        time.sleep(0.1)
        break

    print(f"Received: {data}")
