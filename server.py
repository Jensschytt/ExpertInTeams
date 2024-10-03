import socket
import pickle
import struct
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

HOST = ''
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()

def recv_all(conn, size):
    data = b''
    while len(data) < size:
        try:
            packet = conn.recv(size - len(data))
            if not packet:
                return None  # Connection closed
            data += packet
        except socket.error as e:
            print(f"Socket error: {e}")
            return None
    return data

# Fixed message size for the frame length (4 bytes)
payload_size = struct.calcsize("!I")

try:
    while True:
        # Retrieve exactly 4 bytes of the message size
        packed_msg_size = recv_all(conn, payload_size)
        if packed_msg_size is None:
            print("Connection closed by client")
            break

        # Now safely unpack the message size (4-byte unsigned integer)
        msg_size = struct.unpack("!I", packed_msg_size)[0]
        print(f"Receiving frame of size: {msg_size}")

        # Retrieve the frame data of size 'msg_size'
        frame_data = recv_all(conn, msg_size)
        if frame_data is None:
            print("Connection closed or error receiving frame data")
            break

        # Extract frame and handle exceptions
        try:
            frame = pickle.loads(frame_data)
        except Exception as e:
            print(f"Error while deserializing frame: {e}")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, verbose=False)

        # Get detected labels from the result
        for result in results:
            labels = result.names  # Accessing label names
            detected_classes = result.boxes.cls

            # Convert class indexes to labels and print
            for class_index in detected_classes:
                print(f"Detected object: {labels[int(class_index)]}")

        # Visualize the results (bounding boxes, labels, etc.)
        annotated_frame = results[0].plot()

        # Display the frame with detections
        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)
        cv2.waitKey(1)

except Exception as e:
    print(f"Server error: {e}")
finally:
    print("Closing connection")
    conn.close()