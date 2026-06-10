import json
import time
import math
import random

DATA_FILE = 'face_data.json'
CAM_WIDTH, CAM_HEIGHT = 640, 480

def main():
    print(f"Uruchomiono symulator kamery. Zapis danych do {DATA_FILE}...")
    t = 0.0
    try:
        while True:
            visible = True if random.random() > 0.05 else False
            
            x = (CAM_WIDTH / 2) + math.sin(t) * (CAM_WIDTH / 3)
            y = (CAM_HEIGHT / 2) + math.cos(t * 0.7) * (CAM_HEIGHT / 4)
            
            data = {
                "face_visible": visible,
                "identity": "TestUser",
                "emotion": "Neutral",
                "bbox_center": [int(x), int(y)]
            }
            
            with open(DATA_FILE, 'w') as f:
                json.dump(data, f)
            
            t += 0.1
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nZakończono symulację.")

if __name__ == "__main__":
    main()