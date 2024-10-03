
import pickle
import socket
import struct
import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO

app = Flask(__name__)

# Initialize the YOLOv8 model
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

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")

def generate_frames():
    while True:

        
        # Retrieve message size
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)

        
        # Capture frame-by-frame
        success, frame = video_capture.read()
        if not success:
            print("Error: Failed to capture image.")
            break
        else:
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

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Yield the frame in the required format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
