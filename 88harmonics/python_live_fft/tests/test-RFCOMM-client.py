
from bluetooth import *

# Create the client socket
client_socket=BluetoothSocket( RFCOMM )

client_socket.connect(("A4:D1:8C:D2:52:39", 3))

client_socket.send("Hello World")

print "Finished"

client_socket.close()




