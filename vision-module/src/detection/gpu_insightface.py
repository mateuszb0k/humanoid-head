import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
import numpy as np
import sqlite3
import pickle
import time
import json
import threading
from collections import deque, Counter
from insightface.app import FaceAnalysis
from keras.models import load_model

DB_FILE = 'faces.db'
OUT_JSON = 'faces.json'
MODEL_PATH = 'model.keras'

DETECTION_SIZE = (640, 640)
DETECTION_THRESHOLD = 0.6
EMOTION_INPUT_SIZE = (224, 224)
EMOTION_SKIP_FRAMES = 3
MATCH_THRESHOLD = 1.0
DEAD_ZONE = 0.25
LAST_FRAMES = 10
USER_LIMIT = 200
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class VideoStream:
    """
    A class to read video frames from a camera in a background thread.
    This helps to keep the main program running fast without waiting for the camera.
    """

    def __init__(self, src=0):
        """
        Sets up the camera connection and sets the video resolution.
        """
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        """
        Starts the background thread to continuously read frames.
        """
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        """
        Keeps grabbing new frames from the camera until stopped.
        """
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """
        Returns the newest video frame.
        """
        return self.frame

    def stop(self):
        """
        Stops the camera thread and releases the camera.
        """
        self.stopped = True
        self.stream.release()


class FaceSystem:
    """
    The main class that handles face detection, user recognition, and emotion tracking.
    """

    def __init__(self):
        """
        Prepares the database, loads the AI models for faces and emotions,
        and sets up variables for tracking history and time.
        """
        self.db = self._initialize_db()
        self.db_names, self.db_vecs = self._load_users_from_db()

        trash = open(os.devnull, 'w')
        old_out = sys.stdout
        sys.stdout = trash
        self.analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.analyzer.prepare(ctx_id=0, det_size=DETECTION_SIZE, det_thresh=DETECTION_THRESHOLD)
        self.emotion_net = load_model(MODEL_PATH)
        sys.stdout = old_out

        self.hist = {}
        self.last_emo = {}
        self.f_count = 0
        self.last_tick = time.time()
        self.is_saving = False

    def _initialize_db(self):
        """
        Creates the database file and the users table if they do not exist.
        """
        db = sqlite3.connect(DB_FILE, check_same_thread=False)
        cur = db.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           name
                           TEXT,
                           encoding
                           BLOB
                       )''')
        db.commit()
        return db

    def _load_users_from_db(self):
        """
        Loads all saved user names and their face features from the database.
        """
        cur = self.db.cursor()
        cur.execute("SELECT name, encoding FROM users ORDER BY id ASC")
        rows = cur.fetchall()
        if not rows: return [], np.array([])
        names = [r[0] for r in rows]
        vecs = np.array([pickle.loads(r[1]) for r in rows])
        return names, vecs

    def _save_user_async(self, vector):
        """
        Asks the user for a name in the console and saves their face data to the database.
        Runs in the background so the video doesn't freeze.
        """
        print("\nEnter name in console and press Enter: ")
        name = sys.stdin.readline().strip()

        if name:
            cur = self.db.cursor()
            cur.execute("SELECT COUNT(*) FROM users")
            if cur.fetchone()[0] >= USER_LIMIT:
                cur.execute("DELETE FROM users WHERE id = (SELECT MIN(id) FROM users)")
            cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, pickle.dumps(vector)))
            self.db.commit()
            self.db_names, self.db_vecs = self._load_users_from_db()
            print(f"Saved: {name}")
        else:
            print("No name provided")

        self.is_saving = False

    def save_face_json(self, face_info):
        """
        Saves the current face identities and emotions into a JSON file.
        """
        data = []
        if not face_info:
            data = [{"face_visible": False, "identity": "None", "emotion": "None", "features": []}]
        else:
            for f in face_info:
                data.append({"face_visible": True, "identity": f['identity'], "emotion": f['emotion'],
                             "features": f['features']})

        with open(OUT_JSON, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=4)

    def run(self):
        """
        Starts the main video loop. It reads frames, finds faces, detects emotions,
        draws boxes on the screen, and listens for keyboard commands.
        """
        vs = VideoStream().start()
        time.sleep(1.0)
        print("[S] - Save\n[Q] - Exit")
        self.save_face_json([])

        while True:
            frame = vs.read()
            if frame is None: break
            now = time.time()
            delta_time = now - self.last_tick
            if delta_time > 0:
                fps = 1 / delta_time
            else:
                fps = 0
            self.last_tick = now

            h, w = frame.shape[:2]
            pad_x = int(w * DEAD_ZONE)
            mid_zone = frame[:, pad_x:w - pad_x]

            found = self.analyzer.get(mid_zone)
            faces_now = []
            main_f = None

            if found:
                main_f = max(found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                bx1, by1, bx2, by2 = main_f.bbox.astype(int)
                bx1 += pad_x;
                bx2 += pad_x

                vec = main_f.normed_embedding
                who = "Unknown"
                if len(self.db_vecs) > 0:
                    diffs = np.linalg.norm(self.db_vecs - vec, axis=1)
                    if np.any(diffs < MATCH_THRESHOLD):
                        who = self.db_names[np.argmin(diffs)]

                if who not in self.hist:
                    self.hist[who] = deque(maxlen=LAST_FRAMES)
                    self.last_emo[who] = "None"

                curr_emo = self.last_emo.get(who, "None")
                crop = frame[max(0, by1):max(0, by2), max(0, bx1):max(0, bx2)]

                if crop.size > 0 and self.f_count % EMOTION_SKIP_FRAMES == 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    res = cv2.resize(rgb, EMOTION_INPUT_SIZE)
                    tensor = np.expand_dims(res.astype('float32') / 255.0, 0)
                    preds = self.emotion_net({'input_layer_1': tensor}, training=False).numpy()
                    self.hist[who].append(EMOTIONS[np.argmax(preds)])
                    curr_emo = Counter(self.hist[who]).most_common(1)[0][0]
                    self.last_emo[who] = curr_emo

                faces_now.append({"identity": who, "emotion": curr_emo, "features": vec.tolist()})

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"{who} - {curr_emo}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

            self.save_face_json(faces_now)
            self.f_count += 1

            cv2.rectangle(frame, (pad_x, 0), (w - pad_x, h), (50, 50, 50), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.is_saving:
                cv2.putText(frame, "Enter name in console", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('FaceAI', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not self.is_saving:
                if main_f is not None:
                    self.is_saving = True
                    threading.Thread(target=self._save_user_async, args=(main_f.normed_embedding,), daemon=True).start()
                else:
                    print("No face in zone")
            elif key == ord('q'):
                break

        vs.stop()
        cv2.destroyAllWindows()
        self.db.close()


if __name__ == "__main__":
    FaceSystem().run()

"""
After milestone 1 the system architecture
will transmit from continous streaming transmission
to event-based model to optimize resource usage and
data flow for the conversation processing unit.
The face recognition and emotion detection will be
triggered by specific events, such as a user entering
the frame or a significant change in facial expression,
rather than processing every frame continuously.
This will allow the system to focus on relevant
moments and reduce unnecessary computations."""