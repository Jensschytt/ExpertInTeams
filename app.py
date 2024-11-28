from flask import Flask, Response, render_template, jsonify
import socket
import pickle
import struct
import threading
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

# YOLOv8 Model
model = YOLO('yolov8n.pt')

# Global variables
latest_processed_frame = None
frame_lock = threading.Lock()
detected_objects_count = {"person": 0, "keyboard": 0, "remote": 0}

# Function to handle socket connection and receive frames from client.py
def receive_frames():
    global latest_processed_frame, detected_objects_count
    HOST = ''
    PORT = 8089
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print('Socket now listening for incoming connections')

    conn, addr = s.accept()
    print(f"Connected to: {addr}")

    payload_size = struct.calcsize("!I")
    frame_count = 0  # For frame-skipping

    while True:
        try:
            # Receive message size
            data = b''
            while len(data) < payload_size:
                packet = conn.recv(payload_size - len(data))
                if not packet:
                    return
                data += packet

            packed_msg_size = data
            msg_size = struct.unpack("!I", packed_msg_size)[0]

            # Receive frame data
            data = b''
            while len(data) < msg_size:
                packet = conn.recv(msg_size - len(data))
                if not packet:
                    return
                data += packet

            frame_data = pickle.loads(data)

            # Skip frames for efficiency
            frame_count += 1
            if frame_count % 3 != 0:  # Process every third frame
                continue

            # Resize frame for faster processing
            frame_data = cv2.resize(frame_data, (640, 480))

            # Run YOLOv8 inference on the frame
            results = model(frame_data, verbose=False)

            # Reset count for each detection frame
            detected_objects_count = {"person": 0, "keyboard": 0, "remote": 0}
            for result in results:
                for class_index in result.boxes.cls:
                    class_name = result.names[int(class_index)]
                    if class_name in detected_objects_count:
                        detected_objects_count[class_name] += 1

            # Annotate the frame
            annotated_frame = results[0].plot()
            _, processed_buffer = cv2.imencode('.jpg', annotated_frame)

            with frame_lock:
                latest_processed_frame = processed_buffer.tobytes()

        except Exception as e:
            print(f"Error receiving frame: {e}")
            break

    conn.close()

# Function to generate video feed
def generate_processed_feed():
    global latest_processed_frame
    while True:
        with frame_lock:
            if latest_processed_frame is None:
                continue
            frame = latest_processed_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/processed_feed')
def processed_feed():
    return Response(generate_processed_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/object_statistics')
def object_statistics():
    return jsonify(detected_objects_count)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socket_thread = threading.Thread(target=receive_frames, daemon=True)
    socket_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False)
