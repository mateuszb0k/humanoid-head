import mediapipe as mp
import cv2
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np

emotions_model_path = "./best_71.keras"
model_path = "./blaze_face_light.tflite"
detection_results = []
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

emotions_model = tf.keras.models.load_model(emotions_model_path, compile=False)


# Create a face detector instance with the live stream mode:
def result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global detection_results
    detection_results = result.detections

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result)

#cap = cv2.VideoCapture('http://localhost:4747/mjpegfeed')
cap = cv2.VideoCapture(0)

with FaceDetector.create_from_options(options) as detector:
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time()*1000)

            detector.detect_async(mp_image, timestamp_ms)
            if detection_results:
                for detection in detection_results:
                    bbox = detection.bounding_box

                    x=bbox.origin_x
                    y=bbox.origin_y
                    w = bbox.width
                    h = bbox.height

                    new_y = y-int(h*0.25)

                    start = (x,new_y)
                    end = (x+w,y+h)
                    color = (0,0,255)
                    thickness = 3
                    cv2.rectangle(frame, start, end, color, thickness)
                    start_y,end_y = max(0,start[1]),min(frame.shape[0],end[1])
                    start_x,end_x = max(0, start[0]), min(frame.shape[1], end[0])
                    if start_y >= end_y or start_x >= end_x:
                        continue
                    cut_frame = rgb_frame[start_y:end_y,start_x:end_x]
                    res_frame = cv2.resize(cut_frame,(224,224))
                    input = np.expand_dims(res_frame,axis=0)
                    input = input.astype('float32')/255.0
                    prediction = emotions_model.predict(input,verbose=0)
                    idx = np.argmax(prediction[0])
                    emotion = emotions[idx]
                    pos = (x,new_y-10)
                    cv2.putText(frame,emotion,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv2.imshow('temp', cv2.resize(frame, (1920, 1080)))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except cv2.error:
            break

cap.release()
cv2.destroyAllWindows()