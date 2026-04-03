import os
import cv2
import face_recognition
import numpy as np
import time
import sqlite3
import pickle
import threading
import sys
from collections import deque, Counter
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

DatabasePath = 'faces.db'
Factor = 0.25
ModelPath = 'model.keras'
Size = (224, 224)
Skip = 3
Dictionary = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

app = Flask(__name__)

#Initialize dataBase
def DataBase_initialize():
    db = sqlite3.connect(DatabasePath, check_same_thread=False)
    db.cursor().execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, encoding BLOB)")
    db.commit()
    return db

db = DataBase_initialize()
last = None # last detected face vector
saved = False  #flag for saving

def get_current(): #get all saved faces in db
    cur = db.cursor()
    cur.execute("SELECT name, encoding FROM users")
    rows = cur.fetchall()
    return [r[0] for r in rows], [pickle.loads(r[1]) for r in rows]

# load all saved faces to memory on the begining
saved_names, saved_vecs = get_current()

# function to get name for the new detected face from cmd
def console_listener():
    global saved, saved_names, saved_vecs
    print("\nPresizes 's' to save new face in dataBase")
    while True:
        cmd = sys.stdin.readline().strip().lower()
        if cmd == 's':
            if last is not None:
                saved = True
                print("Give name for current face:")
                name = sys.stdin.readline().strip()
                if name:
                    cur = db.cursor()
                    cur.execute("INSERT INTO users (name, encoding) VALUES (?, ?)", (name, pickle.dumps(last)))
                    db.commit()
                    saved_names, saved_vecs = get_current()
                    print(f"Saved: {name}")
                saved = False
            else:
                print("Error, i don't see any face in the camera")

# cmd listener works in background
threading.Thread(target=console_listener, daemon=True).start()


model = load_model(ModelPath)#load emotions model
#configure camera for RPI 
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

hist = {} #emotions history for majority filter
last_emo = {} 

#frame handling function
def generate_frames():
    global hist, last_emo, last, saved_names, saved_vecs
    last_tick = time.time()
    f_count = 0 

    while True:
        rgb = picam2.capture_array() # get frame from camera in rgb
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # convert for bgr for saving
        
        # calculate frames per second
        start = time.time()
        fps = 1 / (start - last_tick) if (start - last_tick) > 0 else 0
        last_tick = start
        
        input_frame = cv2.resize(rgb, (0, 0), fx=Factor, fy=Factor)
        founded_face = face_recognition.founded_face(input_frame, model="hog")# finding face using hog
        face_encodings = face_recognition.face_encodings(input_frame, founded_face)
        
        for (top, right, bottom, left), encoding in zip(founded_face, face_encodings):
            last = encoding 
            
            indetified = "Unkstartn"
            # compare acttual vector with db. Tolerance =0.5 (lower more strict)
            if len(saved_vecs) > 0:
                matches = face_recognition.compare_faces(saved_vecs, encoding, tolerance=0.5)
                if True in matches:
                    indetified = saved_names[matches.index(True)]

            #scale coordinates back to original 
            t, r, b, l = int(top/Factor), int(right/Factor), int(bottom/Factor), int(left/Factor)
            
            if indetified not in hist:
                hist[indetified] = deque(maxlen=10)
                last_emo[indetified] = "None"
                
            curr_emo = last_emo[indetified]
            cutted = rgb[max(0, t):max(0, b), max(0, l):max(0, r)] # cutt face from the frame
            
            """
            we scale the size to fit the model, 
            extract the most common emotions from 10 frames
            """
            if cutted.size > 0 and f_count % Skip == 0:
                resize = cv2.resize(cutted, Size)
                tensor = np.expand_dims(resize.astype('float32') / 255.0, 0)
                prediciton = model({'input_layer_1': tensor}, training=False).numpy()
                hist[indetified].append(Dictionary[np.argmax(prediciton)])
                curr_emo = Counter(hist[indetified]).most_common(1)[0][0]
                last_emo[indetified] = curr_emo

            cv2.rectangle(bgr, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(bgr, f"{indetified} - {curr_emo}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        f_count += 1
        cv2.putText(bgr, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if saved:
            cv2.putText(bgr, "SAVING MODE - CHECK CONSOLE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret,buffer = cv2.imencode('.jpg', bgr)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# HTML page with live video player 
@app.route('/')
def index():
    return render_template_string('<html><body style="background:#121212; color:white; text-align:center;">'
                                  '<h1>Uncanny Head - HOG + Save</h1>'
                                  '<img src="{{ url_for(\'video_feed\') }}" style="width:80%;">'
                                  '</body></html>')

# MJPEG (Motion JPEG) streaming endpoint
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)