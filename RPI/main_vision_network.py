import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF logs before importing keras
import cv2
import numpy as np
import sqlite3
import pickle
import random
import time
import threading
import sys
import requests
from collections import deque, Counter
from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
from keras.models import load_model
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from face_tracker import move
from adafruit_extended_bus import ExtendedI2C as I2C

DB_FILE = 'faces.db'
DETECTOR_MODEL = 'face_detection_yunet_2023mar.onnx'
RECOGNIZER_MODEL = 'face_recognition_sface_2021dec.onnx'
EMOTION_MODEL = 'model.keras'
MATCH_THRESHOLD = 0.28
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_SKIP_FRAMES = 2
CAM_WIDTH = 1024
CAM_HEIGHT = 864


# Dict emocji do nasladowania przez robota
face_emo = {
    "happy": [
        [(11, 50), (12, 95), (15, 10), (16, 25)],
        [(10, 45)],
        [(19, 130), (18, 70)],
        [(20, 25), (17, 130)],
        [(3, 80), (8, 35), (5, 50), (6, 50)],
        [(2, 100), (4, 30), (7, 110), (9, 10)]],
    "sad": [
        [(11, 25), (12, 60), (15, 50), (16, 70)],
        [(10, 45)],
        [(19, 140), (18, 70)],
        [(20, 70), (17, 70)],
        [(3, 20), (8, 85), (5, 15), (6, 100)],
        [(2, 40), (4, 50), (7, 30), (9, 60)]
    ],
    "surprise": [
        [(11, 0), (12, 110), (15, 0), (16, 90)],
        [(10, 65)],
        [(19, 130), (18, 70)],
        [(20, 20), (17, 130)],
        [(3, 20), (8, 85), (5, 90), (6, 50)],
        [(2, 65), (4, 60), (7, 50), (9, 40)]
    ],
    "neutral": [
        [(11, 25), (12, 70), (15, 40), (16, 70)],
        [(10, 45)],
        [(19, 110), (18, 90)],
        [(20, 50), (17, 110)],
        [(3, 20), (8, 67), (5, 90), (6, 47)],
        [(2, 65), (4, 60), (7, 50), (9, 40)]
    ]
}

# TODO sprawdzić zakresy czy są poprawne
# narzucenie limitów na poszczególne serwa
servos_config = {
    1: {'driver': 0, 'pin': 1, 'min': 100, 'max': 135, 'wlaczone': True},
    2: {'driver': 0, 'pin': 2, 'min': 50, 'max': 65, 'wlaczone': True},
    3: {'driver': 0, 'pin': 3, 'min': 20, 'max': 80, 'wlaczone': True},
    4: {'driver': 0, 'pin': 4, 'min': 50, 'max': 70, 'wlaczone': True},
    5: {'driver': 0, 'pin': 5, 'min': 15, 'max': 50, 'wlaczone': True},
    6: {'driver': 0, 'pin': 6, 'min': 45, 'max': 110, 'wlaczone': True},
    7: {'driver': 0, 'pin': 7, 'min': 40, 'max': 60, 'wlaczone': True},
    8: {'driver': 0, 'pin': 8, 'min': 35, 'max': 85, 'wlaczone': True},
    9: {'driver': 0, 'pin': 9, 'min': 10, 'max': 40, 'wlaczone': True},
    10: {'driver': 0, 'pin': 10, 'min': 50, 'max': 70, 'wlaczone': True},
    11: {'driver': 1, 'pin': 0, 'min': 0, 'max': 60, 'wlaczone': True},
    12: {'driver': 1, 'pin': 1, 'min': 30, 'max': 100, 'wlaczone': True},
    13: {'driver': 1, 'pin': 2, 'min': 40, 'max': 120, 'wlaczone': True, 'center': 60},
    14: {'driver': 1, 'pin': 3, 'min': 50, 'max': 110, 'wlaczone': True, 'center': 100},
    15: {'driver': 1, 'pin': 4, 'min': 10, 'max': 80, 'wlaczone': True},
    16: {'driver': 1, 'pin': 5, 'min': 25, 'max': 70, 'wlaczone': True},
    17: {'driver': 1, 'pin': 6, 'min': 75, 'max': 155, 'wlaczone': True},
    18: {'driver': 1, 'pin': 7, 'min': 70, 'max': 160, 'wlaczone': True},
    19: {'driver': 1, 'pin': 8, 'min': 80, 'max': 170, 'wlaczone': True},
    20: {'driver': 1, 'pin': 9, 'min': 20, 'max': 80, 'wlaczone': True}
}

try:
    i2c = I2C(4)
    driverSO = PCA9685(i2c, address=0x40)
    driverR = PCA9685(i2c, address=0x41)
    driverSO.frequency = 50
    driverR.frequency = 50

except Exception as e:
    print(f"Błąd inicjalizacji I2C (czy testujesz bez sprzętu?): {e}")
    driverR = None
    driverSO = None

servo_objects = {}
if driverSO and driverR:
    for servo_id, cfg in servos_config.items():
        drv = driverSO if cfg['driver'] == 0 else driverR
        servo_objects[servo_id] = servo.Servo(drv.channels[cfg['pin']], min_pulse=730, max_pulse=2930)

app = Flask(__name__)

#mapowanie pikseli na kąty na serwa
def map_servo_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

_servo_lock = threading.Lock()

# funkcja do ustawiania docelowych kątów na serwa
def set_servo_angle(servo_id, angle):
    cfg = servos_config.get(servo_id)
    if not cfg or not cfg['wlaczone']:
        print("wylaczone")
        return
    if servo_id not in servo_objects:
        print("nie ma serwa")
        return
    limit_min = min(cfg['min'], cfg['max'])
    limit_max = max(cfg['min'], cfg['max'])
    safe_angle = max(limit_min, min(angle, limit_max))
    # print(safe_angle,angle) if servo_id == 10 else 0
    with _servo_lock:
        servo_objects[servo_id].angle = safe_angle
    if servo_id in (1, 10):
            other_id = 10 if servo_id == 1 else 1
            cfg1 = servos_config[servo_id]
            cfg2 = servos_config[other_id]
            ratio = (safe_angle - cfg1['min']) / (cfg1['max'] - cfg1['min'])
            mirror_angle = max(cfg2['min'], min(round(cfg2['max'] - ratio * (cfg2['max'] - cfg2['min'])), cfg2['max']))
            servo_objects[other_id].angle = mirror_angle

# funkcja do podążania oczami za użytkownikiem 
def track_face(x, y):
    target_x = map_servo_value(x, 100, CAM_WIDTH, servos_config[14]['max'], servos_config[14]['min'])
    set_servo_angle(14, target_x)
    target_y = map_servo_value(y,100, CAM_HEIGHT, servos_config[13]['max'], servos_config[13]['min'])
    set_servo_angle(13, target_y-55)


class FaceSystem:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (CAM_WIDTH, CAM_HEIGHT), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
        self.emotion_net = load_model(EMOTION_MODEL)
        self.db = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.init_db()
        self.db_names, self.db_vecs = self.load_users()
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (CAM_WIDTH, CAM_HEIGHT)})
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
        self.mouth_angle = 0

        self.last_blink_time = 0
        self.blink_interval = 5.0
        self.is_blinking = False
        self.blink_start_time = 0

        self.emotion_queue = deque(maxlen=1)

        self.is_animating = False

        # thready do poszczególnych funkcjonalności robota
        threading.Thread(target=self.move_eyes,     daemon=True).start()
        threading.Thread(target=self.move_mouth,    daemon=True).start()
        threading.Thread(target=self.emotion_worker, daemon=True).start()


    def init_db(self):
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

    def cleanup_db(self):
        cur = self.db.cursor()
        cur.execute("DELETE FROM users WHERE created_at <= datetime('now', '-7 days')")
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        if count > 200:
            limit = count - 200
            cur.execute(f"DELETE FROM users WHERE id IN (SELECT id FROM users ORDER BY created_at ASC LIMIT {limit})")
        self.db.commit()

    def load_users(self):
        cur = self.db.cursor()
        cur.execute("SELECT name, encoding FROM users")
        rows = cur.fetchall()
        if not rows:
            return [], []
        names = [r[0] for r in rows]
        vecs = [pickle.loads(r[1]) for r in rows]
        return names, vecs

    # mruganie oczami robota
    def eye_blink(self):
        current_time = time.time()
        if not self.is_blinking and (current_time - self.last_blink_time) > self.blink_interval:
            self.is_blinking = True
            self.blink_start_time = current_time
            set_servo_angle(12, 35)
            set_servo_angle(11, 75)
            set_servo_angle(15, 85)
            set_servo_angle(16, 30)
        if self.is_blinking and (current_time - self.blink_start_time) > 0.150:
            set_servo_angle(12, 110)
            set_servo_angle(11, 0)
            set_servo_angle(15, 25)
            set_servo_angle(16, 85)
            self.is_blinking = False
            self.last_blink_time = current_time
            self.blink_interval = random.uniform(1.5, 5.0)

    # funkcja wspópracująca z podążaniem oczami za użytkownikiem 
    def move_eyes(self):
        while True:
            self.eye_blink()
            # identity = self.locked_identity
            if self.face_visible:
                bbox = self.bbox_cent
                if bbox is not None:
                    track_face(bbox[0], bbox[1])
            else:
                set_servo_angle(14,80)
                set_servo_angle(13, 60)

            time.sleep(0.07)  

    # poruszanie ustami przy mowie
    def move_mouth(self):
        last_angle = -1
        l10, u10 = 45.0, 80.0
        # l1,  u1  = 100.0, 135.0
        while True:
            if not self.is_animating:
                angle = float(self.mouth_angle)
                angle = l10 + (u10 - l10) * angle

                if angle != last_angle:
                    set_servo_angle(10, angle)
                    last_angle = angle
            time.sleep(0.05)

    # worker do naśladowania emocji 
    def emotion_worker(self):
        while True:
            if self.emotion_queue:
                emotion = self.emotion_queue.popleft()

                if self.locked_identity == "Unknown" and emotion not in (None, "None"):
                    self.is_animating = True
                    self.animate_emotion(emotion)
                    self.is_animating = False

            time.sleep(0.05)

    def map_emotion(self, raw_emotion):
        if raw_emotion == 'fear':
            return 'surprise'
        if raw_emotion in ('angry', 'disgust'):
            return 'sad'
        return raw_emotion

    def animate_emotion(self, raw_emotion):
        emotion = self.map_emotion(raw_emotion)
        steps = face_emo.get(emotion)
        if not steps:
            return
        for step in steps:
            for servo_id, angle in step:
                set_servo_angle(servo_id, angle)
        time.sleep(0.04)

    def run_console_listener(self):
        print("Save face: Type 's' + Enter here")
        while True:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 's':
                if self.last_feat is not None:
                    self.cleanup_db()
                    self.is_saving = True
                    print("\nFace detected. Enter person's name:")
                    name = sys.stdin.readline().strip()
                    if name:
                        cur = self.db.cursor()
                        cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)",
                                    (name, pickle.dumps(self.last_feat)))
                        self.db.commit()
                        self.db_names, self.db_vecs = self.load_users()
                        print(f"Saved: {name}\n")
                    self.is_saving = False
                else:
                    print("No face detected in frame\n")

    def generate_frames(self):
        last_tick = time.time()
        while True:
            frame = self.picam2.capture_array()
            frame = cv2.flip(frame, -1)	
            now = time.time()
            fps = 1 / (now - last_tick) if (now - last_tick) > 0 else 0
            last_tick = now

            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            payload = {
                "face_visible": False,
                "identity": "None",
                "emotion": "None",
                "bbox_center": [0, 0]
            }

            if faces is not None:
                main_face = max(faces, key=lambda f: f[2] * f[3])
                coords = main_face[:4].astype(int)
                bbox_center = [int(coords[0] + coords[2] / 2), int(coords[1] + coords[3] / 2)]
                self.bbox_cent = bbox_center
                self.face_visible = True

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

                if not self.is_session_active or current_identity != self.locked_identity or (
                        current_identity == "Unknown" and self.f_count % 15 == 0):
                    self.last_feat = feat

                    crop = frame[max(0, coords[1]):coords[1] + coords[3],
                                 max(0, coords[0]):coords[0] + coords[2]]
                    curr_emo = "neutral"
                    if crop.size > 0:
                        res = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
                        tensor = np.expand_dims(res.astype('float32') / 255.0, 0)
                        preds = self.emotion_net({'input_layer_1': tensor}, training=False).numpy()
                        curr_emo = EMOTIONS[np.argmax(preds)]

                    self.locked_identity = current_identity
                    self.locked_emotion = curr_emo
                    self.is_session_active = True

                    if current_identity == "Unknown" and curr_emo not in (None, "None"):
                        self.emotion_queue.append(curr_emo)

                self.face_lost_time = None
                payload["face_visible"] = True
                payload["identity"] = self.locked_identity
                payload["emotion"] = self.locked_emotion
                payload["bbox_center"] = bbox_center

                color = (0, 165, 255) if self.locked_identity == "Unknown" else (0, 255, 0)
                cv2.rectangle(frame, (coords[0], coords[1]),
                              (coords[0] + coords[2], coords[1] + coords[3]), color, 2)
                label = f"{self.locked_identity} - {self.locked_emotion}"
                cv2.putText(frame, label, (coords[0], coords[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                self.face_visible = False
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
threading.Thread(target=vision_sys.run_console_listener, daemon=True).start()


@app.route('/')
def index():
    return render_template_string('''
        <html>
          <head><title>Uncanny Head AI</title></head>
          <body style="background: #000; color: #fff; text-align: center; font-family: sans-serif;">
            <h1 style="color: #4caf50;">Uncanny Head - Vision Module (YuNet)</h1>
            <p>Status: Hardware Validated (RPi 5)</p>
            <div style="position: relative; display: inline-block;">
                <img src="{{ url_for('video_feed') }}" style="width: 85%; border: 4px solid #333; border-radius: 12px;">
            </div>
            <p style="color: #888;"><i>To save a face, use the SSH terminal and press \'s\'.</i></p>
          </body>
        </html>
    ''')


@app.route('/video_feed')
def video_feed():
    return Response(vision_sys.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/save_name', methods=['POST'])
def save_name():
    data = request.json
    new_name = data.get('identity', "Unknown")
    if vision_sys.last_feat is not None:
        cur = vision_sys.db.cursor()
        vision_sys.cleanup_db()
        cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)",
                    (new_name, pickle.dumps(vision_sys.last_feat)))
        vision_sys.db.commit()
        vision_sys.db_names, vision_sys.db_vecs = vision_sys.load_users()
    return jsonify({"status": "ok"}), 200


@app.route('/api/mouth_status', methods=['POST'])
def change_mouth_status():
    data = request.json
    vision_sys.mouth_angle = data.get("mouth_status")
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
