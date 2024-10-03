from flask import Flask, Response, render_template
import socket
import pickle
import struct
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)

# YOLOv8 Model
model = YOLO('yolov8n.pt')

# Shared variable to hold the latest processed frame
latest_processed_frame = None
frame_lock = threading.Lock()

# Function to handle socket connection and receive frames from client.py
def receive_frames():
    global latest_processed_frame
    HOST = ''  # Listen on all interfaces
    PORT = 8089
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow address reuse
    s.bind((HOST, PORT))
    s.listen(1)
    print('Socket now listening for incoming connections')

    conn, addr = s.accept()
    print(f"Connected to: {addr}")

    payload_size = struct.calcsize("!I")

    while True:
        try:
            # Receive message size
            data = b''
            while len(data) < payload_size:
                packet = conn.recv(payload_size - len(data))
                if not packet:
                    return  # Connection closed
                data += packet

            packed_msg_size = data
            msg_size = struct.unpack("!I", packed_msg_size)[0]

            # Receive frame data
            data = b''
            while len(data) < msg_size:
                packet = conn.recv(msg_size - len(data))
                if not packet:
                    return  # Connection closed
                data += packet

            # Deserialize frame using pickle
            frame_data = pickle.loads(data)

            # Run YOLOv8 inference on the frame
            results = model(frame_data, verbose=False)

            # Annotate the frame
            annotated_frame = results[0].plot()

            # Encode the processed frame in JPEG format
            _, processed_buffer = cv2.imencode('.jpg', annotated_frame)

            # Store the processed frame for serving
            with frame_lock:
                latest_processed_frame = processed_buffer.tobytes()

        except Exception as e:
            print(f"Error receiving frame: {e}")
            break

    conn.close()

# Flask endpoint to serve the processed video feed
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

# Serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Start the socket thread only if this is the main module and not a reloader process
if __name__ == '__main__':
    # Avoid running the socket thread twice when Flask's reloader is active
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        socket_thread = threading.Thread(target=receive_frames, daemon=True)
        socket_thread.start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)