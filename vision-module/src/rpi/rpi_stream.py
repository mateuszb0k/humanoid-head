"""
   sudo apt install libcamera-v4l2 v4l-utils libcap-dev -y

   python -m venv --system-site-packages .venv

   pip install flask opencv-python picamera2
"""

from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from picamera2 import Picamera2

app = Flask(__name__)

# initializing Picamera2 module - Sensor IMX219 <=> RPi Camera v2
try:
    picam2 = Picamera2()
    # 640x480
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("picamera2 initialized successfully")
except Exception as error:
    print(f"failed to start camera: {error}")


def generate_frames():
    """
    Continuously captures video frames from the camera.
    It changes the color format to display correctly and packages
    the images one by one to create a live video stream.
    """
    while True:
        frame = picam2.capture_array()

        # RGB -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # to jpg format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    """
    Sends a simple HTML webpage to the user, displaying the live video.
    """
    # simple interface to check if it's working
    return render_template_string('''
        <html>
          <head>
            <title>Uncanny Head - Vision Module Live</title>
            <style>
                body { background: #121212; color: #e0e0e0; text-align: center; font-family: sans-serif; }
                .container { margin-top: 50px; }
                img { border: 4px solid #333; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); width: 80%; }
                .status { color: #4caf50; font-weight: bold; }
            </style>
          </head>
          <body>
            <div class="container">
                <h1>Uncanny Head - Vision Module</h1>
                <p>Status: <span class="status">Hardware Validated (RPi 5)</span></p>
                <img src="{{ url_for('video_feed') }}">
                <p><i>Controlled conditions: office lighting, frontal face orientation.</i></p>
            </div>
          </body>
        </html>
    ''')


@app.route('/video_feed')
def video_feed():
    """
    Provides the continuous stream of video frames to the webpage.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)