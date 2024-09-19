import cv2
import numpy as np
import socket
import sys
import pickle
import struct
from ultralytics import YOLO

# Load the YOLOv8 model (e.g., yolov8n for speed)
model = YOLO('yolov8n.pt')
print(model.names)

# Open the webcam
#video_capture = cv2.VideoCapture(0)

cap=cv2.VideoCapture(0)

clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('10.126.99.161',8089))

while True:



    ret,frame=cap.read()

    if not ret:
        print("Failed to grab frame")
        break

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

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Serialize frame
    data = pickle.dumps(frame)

    # Send message length first
    message_size = struct.pack("L", len(data)) ### CHANGED

    # Then data
    clientsocket.sendall(message_size + data)