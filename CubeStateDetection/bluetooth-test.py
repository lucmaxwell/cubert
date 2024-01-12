import socket
import msvcrt
import random

y = b'y'
y_ = b'Y'
b = b'b'
b_ = b'B'
x_ = b'X'
ack = b'a'

messages = [y, y_, b, b_, x_]

client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
client.connect(("40:22:D8:F0:E6:1A", 1))
client.setblocking(False)

print("Connected?")
message = ""

while True:
    if msvcrt.kbhit():
        letter = msvcrt.getch().decode("utf-8")
        message += letter
        
        if letter == "\r":
            client.send(message.encode("utf-8"))
            message = ""

            print()
        print(letter, end="")

    try:
        data = client.recv(4)
        print(f"Received: {data}")

        if data == ack:
            msg = random.choice(messages)
            print(f"Sending: {msg}")
            client.send(msg)
            client.send(b'\r')

    except:
        print("",end="")