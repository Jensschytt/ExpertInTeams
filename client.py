import cv2
import socket
import pickle
import struct

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create a socket to connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_ip = '10.126.58.176'  # Replace with the server's IP address
server_port = 8089  # Ensure it matches the port in app.py

try:
    client_socket.connect((server_ip, server_port))
    print("Connected to the server!")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Serialize frame
        data = pickle.dumps(frame)

        # Send frame size and data
        message_size = struct.pack("!I", len(data))
        try:
            client_socket.sendall(message_size + data)
        except socket.error as e:
            print(f"Socket error while sending data: {e}")
            break

except Exception as e:
    print(f"Failed to connect to server: {e}")
finally:
    client_socket.close()
    cap.release()