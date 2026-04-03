import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress TF logs before importing keras
import cv2
import numpy as np
import sqlite3
import pickle
import time
import threading
import sys
from collections import deque, Counter
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from keras.models import load_model

DB_FILE = 'faces.db'
DETECTOR_MODEL = 'face_detection_yunet_2023mar.onnx'
RECOGNIZER_MODEL = 'face_recognition_sface_2021dec.onnx'
EMOTION_MODEL = 'model.keras'
MATCH_THRESHOLD = 0.36
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_SKIP_FRAMES = 2

app = Flask(__name__)

class FaceSystem:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (640, 480), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        self.emotion_net = load_model(EMOTION_MODEL)
        self.db = sqlite3.connect(DB_FILE, check_same_thread=False)
        self._init_db()
        self.db_names, self.db_vecs = self._load_users()
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        self.last_feat = None
        self.is_saving = False
        self.hist = {}
        self.last_emo = {}
        self.f_count = 0

    def _init_db(self):
        cur = self.db.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding BLOB)")
        self.db.commit()

    def _load_users(self):
        cur = self.db.cursor()
        cur.execute("SELECT name, encoding FROM users")
        rows = cur.fetchall()
        if not rows: 
            return [], []
        names = [r[0] for r in rows]
        vecs = [pickle.loads(r[1]) for r in rows]
        return names, vecs

    def run_console_listener(self):
        print("Save face: Type 's' + Enter here")
        while True:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 's':
                if self.last_feat is not None:
                    self.is_saving = True
                    print("\nFace detected. Enter person's name:")
                    name = sys.stdin.readline().strip()
                    if name:
                        cur = self.db.cursor()
                        cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, pickle.dumps(self.last_feat)))
                        self.db.commit()
                        self.db_names, self.db_vecs = self._load_users()
                        print(f"Saved: {name}\n")
                    self.is_saving = False
                else:
                    print("No face detected in frame\n")

    def generate_frames(self):
        last_tick = time.time() 
        while True:
            frame = self.picam2.capture_array()
            # FPS calculation
            now = time.time()
            fps = 1 / (now - last_tick) if (now - last_tick) > 0 else 0
            last_tick = now
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            status, faces = self.detector.detect(frame)
            if faces is not None:
                # saving largest face for database
                main_face = max(faces, key=lambda f: f[2] * f[3])
                for face in faces:
                    aligned = self.recognizer.alignCrop(frame, face)
                    feat = self.recognizer.feature(aligned)
                    if np.array_equal(face, main_face):
                        self.last_feat = feat
                    # selecting highest score 
                    identity = "Unknown"
                    if len(self.db_vecs) > 0:
                        for idx, db_vec in enumerate(self.db_vecs):
                            score = self.recognizer.match(feat, db_vec, cv2.FaceRecognizerSF_FR_COSINE)
                            if score >= MATCH_THRESHOLD:
                                identity = self.db_names[idx]
                                break
                    coords = face[:4].astype(int)
                    if identity not in self.hist:
                        self.hist[identity] = deque(maxlen=10)
                        self.last_emo[identity] = "Neutral"
                    curr_emo = self.last_emo[identity]
                    # processsing emotion recognition periodically to save CPU
                    if self.f_count % EMOTION_SKIP_FRAMES == 0:
                        crop = frame[max(0, coords[1]):coords[1]+coords[3], 
                                     max(0, coords[0]):coords[0]+coords[2]]
                        if crop.size > 0:
                            res = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
                            tensor = np.expand_dims(res.astype('float32') / 255.0, 0)
                            preds = self.emotion_net({'input_layer_1': tensor}, training=False).numpy()
                            self.hist[identity].append(EMOTIONS[np.argmax(preds)])
                            curr_emo = Counter(self.hist[identity]).most_common(1)[0][0]
                            self.last_emo[identity] = curr_emo
                    color = (0, 255, 0) if identity != "Unknown" else (0, 165, 255)
                    cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                    label = f"{identity} - {curr_emo}"
                    cv2.putText(frame, label, (coords[0], coords[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.f_count += 1
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.is_saving:
                cv2.rectangle(frame, (0, 440), (640, 480), (0, 0, 255), -1)
                cv2.putText(frame, "SAVE MODE", (150, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

vision_sys = FaceSystem()
# background thread for SSH terminal input
threading.Thread(target=vision_sys.run_console_listener, daemon=True).start()

@app.route('/')
def index():
    #simple interface to check if it's working
    return render_template_string('''
        <html>
          <head><title>Uncanny Head AI</title></head>
          <body style="background: #000; color: #fff; text-align: center; font-family: sans-serif;">
            <h1 style="color: #4caf50;">Uncanny Head - Vision Module (YuNet)</h1>
            <p>Status: Hardware Validated (RPi 5)</p>
            <div style="position: relative; display: inline-block;">
                <img src="{{ url_for('video_feed') }}" style="width: 85%; border: 4px solid #333; border-radius: 12px;">
            </div>
            <p style="color: #888;"><i>To save a face, use the SSH terminal and press 's'.</i></p>
          </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(vision_sys.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)