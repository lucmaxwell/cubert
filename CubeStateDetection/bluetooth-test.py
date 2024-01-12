import socket
import msvcrt
import random
import cv2 as cv
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import urllib.request
import scipy.stats as stats

def getImage(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(arr, -1) # 'Load it as it is'
    return img

# Kind of also parameters but not really
basePath = os.getcwd() + "\\CubeStateDetection\\vision\\"
imagesPath = basePath + "images/"
outPath = basePath + "output/"
imageUrl = "http://192.168.4.1/capture"

y = b'y'
y_ = b'Y'
b = b'b'
b_ = b'B'
x_ = b'X'
ACK = b'a'
OK = b'k'

messages = [y, y_, b, b_, x_]

client = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
client.connect(("40:22:D8:F0:E6:1A", 1))
client.setblocking(False)

print("Connected?")
message = ''

while True:
    if msvcrt.kbhit():
        letter = msvcrt.getch().decode("utf-8")
        
        if letter == "\r":
            client.send(message.encode("utf-8"))
            print(f"\nSent '{message}'")
            message = ""
            
        else: 
            message = letter
            print(letter, end="")

    try:
        data = client.recv(1)
        print(f"Received: {data}")

        # if data == ACK:
        #     msg = random.choice(messages)
        #     print(f"Sending: {msg}")
        #     client.send(msg)
        #     client.send(b'\r')

        if data == OK:
            print("Recieved OK, taking image")
            img = getImage(imageUrl)
            cv.imwrite(outPath + 'test.png', img)
            print("",end="")

    except:
        print("",end="")