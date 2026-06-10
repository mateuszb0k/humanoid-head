import os
import sys
import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
import time
import threading

DB_FILE = 'faces.db'
TOLERANCE = 0.5
SCALE_FACTOR = 0.25
INVERSE_SCALE = int(1 / SCALE_FACTOR)

class VideoStream:
    """
    A class to read video frames from a camera using a separate background thread.
    This prevents the camera from slowing down the main face recognition process.
    """
    def __init__(self, src=0):
        """
        Sets up the camera connection and reads the very first frame.
        """
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        """
        Starts the background thread that will continuously update the frames.
        """
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        """
        An endless loop that keeps grabbing the newest frame from the camera
        until the stream is stopped.
        """
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """
        Returns the most recently captured video frame.
        """
        return self.frame

    def stop(self):
        """
        Stops the camera thread and safely releases the camera hardware.
        """
        self.stopped = True
        self.stream.release()

class HogSystem:
    """
    The main system for detecting and recognizing faces.
    It uses the HOG (Histogram of Oriented Gradients) method, which is good for running on a CPU.
    """
    def __init__(self):
        """
        Prepares the system by setting up the database, loading saved users,
        and preparing timers for FPS calculation.
        """
        self.db = self._init_db()
        self.db_names, self.db_vecs = self._get_saved_peeps()
        
        self.last_tick = time.time()
        self.is_saving = False

    def _init_db(self):
        """
        Connects to the local database file. Creates a new table for saving
        users and their face data if it does not exist yet.
        """
        db = sqlite3.connect(DB_FILE, check_same_thread=False)
        cur = db.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, encoding BLOB)''')
        db.commit()
        return db

    def _get_saved_peeps(self):
        """
        Loads all the saved user names and their face encodings from the database
        so the system can recognize them in the video.
        """
        cur = self.db.cursor()
        cur.execute("SELECT name, encoding FROM users ORDER BY id ASC")
        rows = cur.fetchall()
        if not rows: return [], []
        names = [r[0] for r in rows]
        vecs = [pickle.loads(r[1]) for r in rows]
        return names, vecs

    def _async_save_worker(self, encoding):
        """
        Runs in the background when saving a new face. It asks the user to type
        a name in the console, then saves that name and face data to the database.
        """
        print("\n[SAVE] Enter name in console and press Enter: ")
        name = sys.stdin.readline().strip()
        
        if name:
            cur = self.db.cursor()
            cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", 
                        (name, pickle.dumps(encoding)))
            self.db.commit()
            self.db_names, self.db_vecs = self._get_saved_peeps()
            print(f"[OK] Saved to database: {name}")
        else:
            print("[CANCELLED] No name provided.")
        
        self.is_saving = False

    def run(self):
        """
        The main loop of the program. It captures video, shrinks it for faster processing,
        finds faces, tries to recognize who they are, and draws boxes with names on the screen.
        """
        print(">>> HOG SYSTEM (CPU) STARTED")
        print(">>> Controls: [S] - Save person | [Q] - Exit")
        
        vs = VideoStream().start()
        time.sleep(1.0)

        while True:
            frame = vs.read()
            if frame is None: break
            
            now = time.time()
            fps = 1 / (now - self.last_tick) if (now - self.last_tick) > 0 else 0
            self.last_tick = now

            small_frame = cv2.resize(frame,(0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current_main_encoding = None

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                identity = "Unknown"
                
                if current_main_encoding is None:
                    current_main_encoding = face_encoding

                if len(self.db_vecs) > 0:
                    matches = face_recognition.compare_faces(self.db_vecs, face_encoding, tolerance=TOLERANCE)
                    if True in matches:
                        first_match_index = matches.index(True)
                        identity = self.db_names[first_match_index]

                top, right, bottom, left = top * INVERSE_SCALE, right * INVERSE_SCALE, bottom * INVERSE_SCALE, left * INVERSE_SCALE
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, identity, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.is_saving:
                cv2.putText(frame, "TYPE NAME IN CONSOLE...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Face Recognition System (CPU/HOG)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not self.is_saving:
                if current_main_encoding is not None:
                    self.is_saving = True
                    threading.Thread(target=self._async_save_worker, args=(current_main_encoding,), daemon=True).start()
                else:
                    print("No face detected for saving.")
            elif key == ord('q'):
                break

        vs.stop()
        cv2.destroyAllWindows()
        self.db.close()

if __name__ == "__main__":
    HogSystem().run()


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
moments and reduce unnecessary computations.

"""