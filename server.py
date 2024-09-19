import pickle
import socket
import struct
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (e.g., yolov8n for speed)
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

data = b'' ### CHANGED
payload_size = struct.calcsize("L") ### CHANGED




while True:

    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame
    frame = pickle.loads(frame_data)



    # Run YOLOv8 inference on the frame
    results = model(frame, verbose=False)

    # Get detected labels from the result
    for result in results:
        labels = result.names  # Accessing label names
        detected_classes = result.boxes.cls

    # Convert class indexes to labels and print
        for class_index in detected_classes:
            print(f"Detected object: {labels[int(class_index)]}")
    

    print("test")
    # Visualize the results (bounding boxes, labels, etc.)
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # Display
    cv2.waitKey(1)