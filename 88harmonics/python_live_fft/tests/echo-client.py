# echoclient.py

from bluetooth import *

HOST = "A4:D1:8C:D2:52:39"  # The remote host
PORT = 3  # Server port

s = BluetoothSocket(RFCOMM)

s.connect((HOST, PORT))

while True:
    message = input('Send:')
    if not message: break
    s.send(message)
    data = s.recv(1024)
    print('Received', data)
s.close()
