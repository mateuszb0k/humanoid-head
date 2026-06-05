import cv2
import sys

def check_camera():
    #0 is our default camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Couldn't initialize the video stream")
        return False

    #read a single frame
    ret, frame = cap.read()
    
    if ret:
        print("Camera initialized successfully")
        # Save a test frame to disk
        cv2.imwrite("test_capture.jpg", frame)
    else:
        print("Couldn't read frame from the camera")

    cap.release()
    return ret

if __name__ == "__main__":
    check_camera()