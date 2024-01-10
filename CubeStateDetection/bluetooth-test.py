import socket
import time

client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
client.connect(("40:22:D8:F0:E6:1A", 1))

print("Connected?")
 
while True:
    client.send("Reverse testing\r\n".encode("utf-8"))
    data = client.recv(9)
    if not data:
        break
    print(f"Received: {data}")
    time.sleep(1)