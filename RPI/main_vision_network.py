import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress TF logs before importing keras
import cv2
import numpy as np
import sqlite3
import pickle
import time
import threading
import sys
import requests
from collections import deque, Counter
from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
from keras.models import load_model
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from face_tracker import move

DB_FILE = 'faces.db'
DETECTOR_MODEL = 'face_detection_yunet_2023mar.onnx'
RECOGNIZER_MODEL = 'face_recognition_sface_2021dec.onnx'
EMOTION_MODEL = 'model.keras'
MATCH_THRESHOLD = 0.28
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_SKIP_FRAMES = 2
CAM_WIDTH = 640
CAM_HEIGHT = 480
servos_config = {
    10: {'driver': 0, 'pin': 10, 'min': 50, 'max': 70, 'wlaczone': True},   # usta
    13: {'pin': 2, 'min': 40, 'max': 120, 'wlaczone': True, 'center': 60},  # oczy góra-dół
    14: {'pin': 3, 'min': 50, 'max': 150, 'wlaczone': True, 'center': 100}  # oczy prawo-lewo
}
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    driverR = PCA9685(i2c, address=0x41)
    driverR.frequency = 50
except Exception as e:
    print(f"Błąd inicjalizacji I2C (czy testujesz bez sprzętu?): {e}")
    driverR = None
servo_objects = {}
if driverR:
    for servo_id, cfg in servos_config.items():
        servo_objects[servo_id] = servo.Servo(driverR.channels[cfg['pin']], min_pulse=730, max_pulse=2930)
app = Flask(__name__)
def map_servo_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def set_servo_angle(servo_id, angle):
    cfg = servos_config.get(servo_id)
    if not cfg or not cfg['wlaczone']:
        return

    limit_min = min(cfg['min'], cfg['max'])
    limit_max = max(cfg['min'], cfg['max'])
    safe_angle = max(limit_min, min(angle, limit_max))
    if driverR:
        servo_objects[servo_id].angle = safe_angle
def track_face(x, y):

    target_x = map_servo_value(x, 0, CAM_WIDTH, servos_config[14]['max'], servos_config[14]['min'])
    set_servo_angle(14, target_x)

    target_y = map_servo_value(y, 0, CAM_HEIGHT, servos_config[13]['max'], servos_config[13]['min'])
    set_servo_angle(13, target_y)
class FaceSystem:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (CAM_WIDTH, CAM_HEIGHT), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        self.emotion_net = load_model(EMOTION_MODEL)
        self.db = sqlite3.connect(DB_FILE, check_same_thread=False)
        self._init_db()
        self.db_names, self.db_vecs = self._load_users()
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (CAM_WIDTH, CAM_HEIGHT)})
        config.transform.hflip = True
        config.transform.vflip = True
        self.picam2.configure(config)
        self.picam2.start()
        self.last_feat = None
        self.is_saving = False
        self.hist = {}
        self.last_emo = {}
        self.f_count = 0
        self.locked_identity = "None"
        self.locked_emotion = "None"
        self.is_session_active = False
        self.face_lost_time = None
        self.id_buffer = deque(maxlen=7)
        self.face_visible = False
        self.bbox_cent = None
        td = threading.Thread(target=self.move_eyes, daemon=True)
        td.start()

    def _init_db(self):
        cur = self.db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                name TEXT, 
                encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()
    def move_eyes(self):
        while True:
            if self.face_visible:
                bbox = self.bbox_cent
                track_face(bbox[0], bbox[1])
                print(f"Śledzę twarz: x={bbox[0]}, y={bbox[1]}")
            else:
                set_servo_angle(14, servos_config[14]['center'])
                set_servo_angle(13, servos_config[13]['center'])
                print("Brak twarzy - powrót do centrum.")

            time.sleep(0.7)

    def _cleanup_db(self):
        cur = self.db.cursor()
        # delete records strictly older than 7 days based on timestamp
        cur.execute("DELETE FROM users WHERE created_at <= datetime('now', '-7 days')")
        # enforce the 200 max users limit (delete oldest if exceeded)
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        if count > 200:
            limit = count - 200
            # subquery to find the oldest IDs to delete
            cur.execute(f"DELETE FROM users WHERE id IN (SELECT id FROM users ORDER BY created_at ASC LIMIT {limit})")
        
        self.db.commit()

    def run_console_listener(self):
        print("Save face: Type 's' + Enter here")
        while True:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 's':
                if self.last_feat is not None:
                    # run cleanup before saving a new face
                    self._cleanup_db() 
                    
                    self.is_saving = True
                    print("\nFace detected. Enter person's name:")
                    name = sys.stdin.readline().strip()
                    if name:
                        cur = self.db.cursor()
                        cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, pickle.dumps(self.last_feat)))
                        self.db.commit()
                        
                        # reload into RAM after DB changes
                        self.db_names, self.db_vecs = self._load_users()
                        print(f"Saved: {name}\n")
                    self.is_saving = False
                else:
                    print("No face detected in frame\n")

    def _load_users(self):
        cur = self.db.cursor()
        cur.execute("SELECT name, encoding FROM users")
        rows = cur.fetchall()
        if not rows: 
            return [], []
        names = [r[0] for r in rows]
        vecs = [pickle.loads(r[1]) for r in rows]
        return names, vecs

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
            payload = {
                "face_visible": False,
                "identity": "None",
                "emotion": "None",
                "bbox_center": [0, 0]
            }
            if faces is not None:
                # saving largest face for database
                main_face = max(faces, key=lambda f: f[2] * f[3])
                coords = main_face[:4].astype(int)
                bbox_center = [int(coords[0] + coords[2]/2), int(coords[1] + coords[3]/2)]
                self.bbox_cent = bbox_center
                self.face_visible = True
                # recognition check on every frame for dynamic switching
                aligned = self.recognizer.alignCrop(frame, main_face)
                feat = self.recognizer.feature(aligned)
                
                raw_identity = "Unknown"
                if len(self.db_vecs) > 0:
                    for idx, db_vec in enumerate(self.db_vecs):
                        score = self.recognizer.match(feat, db_vec, cv2.FaceRecognizerSF_FR_COSINE)
                        if score >= MATCH_THRESHOLD:
                            raw_identity = self.db_names[idx]
                            break
                        
                self.id_buffer.append(raw_identity)
                counts = Counter(self.id_buffer)
                current_identity = counts.most_common(1)[0][0]

                # update if identity changed OR session was inactive
                if not self.is_session_active or current_identity != self.locked_identity:
                    self.last_feat = feat
                    
                    # run emotion model only on identity change or new session
                    crop = frame[max(0, coords[1]):coords[1]+coords[3], 
                                 max(0, coords[0]):coords[0]+coords[2]]
                    curr_emo = "Neutral"
                    if crop.size > 0:
                        res = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
                        tensor = np.expand_dims(res.astype('float32') / 255.0, 0)
                        preds = self.emotion_net({'input_layer_1': tensor}, training=False).numpy()
                        curr_emo = EMOTIONS[np.argmax(preds)]
                    
                    self.locked_identity = current_identity
                    self.locked_emotion = curr_emo
                    self.is_session_active = True

                self.face_lost_time = None
                payload["face_visible"] = True
                payload["identity"] = self.locked_identity
                payload["emotion"] = self.locked_emotion
                payload["bbox_center"] = bbox_center
                color = (0, 165, 255) if self.locked_identity == "Unknown" else (0, 255, 0)
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                label = f"{self.locked_identity} - {self.locked_emotion}"
                cv2.putText(frame, label, (coords[0], coords[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                if self.is_session_active:
                    if self.face_lost_time is None:
                        self.face_lost_time = time.time()
                    elif time.time() - self.face_lost_time > 3.0:
                        self.is_session_active = False
                        self.locked_identity = "None"
                        self.locked_emotion = "None"
                        self.face_lost_time = None
                        self.id_buffer.clear()
                payload["identity"] = self.locked_identity
                payload["emotion"] = self.locked_emotion
            try:
                requests.post("http://192.168.0.143:5000/api/data", json=payload, timeout=0.02)
            except:
                pass
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
@app.route('/api/save_name',methods=['POST'])
def save_name():
    data = request.json
    new_name = data.get('identity',"Unknown")
    if vision_sys.last_feat is not None:
        cur = vision_sys.db.cursor()
        vision_sys._cleanup_db()
        cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (new_name, pickle.dumps(vision_sys.last_feat)))
        vision_sys.db.commit()
        vision_sys.db_names,vision_sys.db_vecs = vision_sys._load_users()
    return jsonify({"status":"ok"}),200

@app.route('/api/mouth_status', methods = ['POST'])
def change_mouth_status():
    data = request.json
    status = data.get("mouth_status")
    return jsonify({"status": "ok"}), 200

def mouth_handle():
    while True:
        set_servo_angle(10, servos_config[10]['max'])
        time.sleep(1)
        set_servo_angle(10, servos_config[10]['min'])
        time.sleep(1)

if __name__ == '__main__':
    td = threading.Thread(target = mouth_handle, daemon = True)
    td.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)