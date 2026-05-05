import time
import json
import os
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

DATA_FILE = 'face_data.json'

CAM_WIDTH = 640
CAM_HEIGHT = 480

try:
    i2c = busio.I2C(board.SCL, board.SDA)
    driverR = PCA9685(i2c, address=0x41)
    driverR.frequency = 50
except Exception as e:
    print(f"Błąd inicjalizacji I2C (czy testujesz bez sprzętu?): {e}")
    driverR = None

# Konfiguracja serw (tylko 13 i 14)
# 'center' to kąt, do którego oczy wracają, gdy twarz znika
servos_config = {
    13: {'pin': 2, 'min': 40, 'max': 120, 'wlaczone': True, 'center': 60},  # oczy góra-dół
    14: {'pin': 3, 'min': 50, 'max': 150, 'wlaczone': True, 'center': 100}  # oczy prawo-lewo
}

servo_objects = {}
if driverR:
    for servo_id, cfg in servos_config.items():
        servo_objects[servo_id] = servo.Servo(driverR.channels[cfg['pin']], min_pulse=730, max_pulse=2930)

def map_value(value, in_min, in_max, out_min, out_max):
    """Przelicza wartość z jednego zakresu na drugi (np. piksele na kąty serwa)"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def set_servo_angle(servo_id, angle):
    """Zabezpieczone przypisanie kąta do serwa"""
    cfg = servos_config.get(servo_id)
    if not cfg or not cfg['wlaczone']:
        return
    
    limit_min = min(cfg['min'], cfg['max'])
    limit_max = max(cfg['min'], cfg['max'])
    safe_angle = max(limit_min, min(angle, limit_max))
    
    if driverR:
        servo_objects[servo_id].angle = safe_angle
    else:
        # print(f"[SYMULACJA] Serwo {servo_id} -> {safe_angle:.1f} stopni")
        pass

def track_face(x, y):
    """Mapuje współrzędne [x,y] na ruch serw"""
    
    target_x = map_value(x, 0, CAM_WIDTH, servos_config[14]['max'], servos_config[14]['min'])
    set_servo_angle(14, target_x)

    target_y = map_value(y, 0, CAM_HEIGHT, servos_config[13]['max'], servos_config[13]['min'])
    set_servo_angle(13, target_y)

def main():
    print(f"Czekam na dane z kamery w pliku '{DATA_FILE}'...")
    last_mod_time = 0
    
    try:
        while True:
            if os.path.exists(DATA_FILE):
                mtime = os.path.getmtime(DATA_FILE)
                
                if mtime != last_mod_time:
                    try:
                        with open(DATA_FILE, 'r') as f:
                            data = json.load(f)
                            
                        if data.get("face_visible"):
                            bbox = data.get("bbox_center", [CAM_WIDTH/2, CAM_HEIGHT/2])
                            track_face(bbox[0], bbox[1])
                            print(f"Śledzę twarz: x={bbox[0]}, y={bbox[1]}")
                        else:
                            set_servo_angle(14, servos_config[14]['center'])
                            set_servo_angle(13, servos_config[13]['center'])
                            print("Brak twarzy - powrót do centrum.")
                            
                        last_mod_time = mtime
                    except (json.JSONDecodeError, KeyError, PermissionError):
                        # Ignoruj błędy, jeśli plik był odczytywany dokładnie w momencie zapisu
                        pass
                        
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\nZakończono działanie.")

if __name__ == "__main__":
    main()