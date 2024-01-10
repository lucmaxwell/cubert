import socket

client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
client.connect(("40:22:D8:F0:E6:18", 2))

print("Connected?")
 
while True:
    client.send("Reverse testing")
    data = client.recv(4)
    if not data:
        break
    print(f"Received: {data}")