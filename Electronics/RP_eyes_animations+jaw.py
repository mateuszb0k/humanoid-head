import time
import random
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

"""
Wymagania przed testem na Raspberry Pi:

W terminalu upewnij się, że I2C jest włączone:
    $ sudo raspi-config

Zainstaluj biblioteki:
    $ pip3 install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor
"""

i2c = busio.I2C(board.SCL, board.SDA)

pca0 = PCA9685(i2c, address=0x40)
pca0.frequency = 50
pca1 = PCA9685(i2c, address=0x41)
pca1.frequency = 50

MIN_PULSE = 732
MAX_PULSE = 2930

servos = []
for i in range(11):
    servos.append(servo.Servo(pca0.channels[i], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE))
for i in range(10):
    servos.append(servo.Servo(pca1.channels[i], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE))

limits = {
    0: (0, 180),      # tongue
    1: (80, 125),     # jaw (zamknięta: 80, otwarta: 125)
    2: (50, 65),      # right upper lip corner
    3: (20, 80),      # right upper lip
    4: (50, 70),      # right lower lip corner
    5: (0, 180),      # right lower lip
    6: (45, 110),     # left lower lip
    7: (40, 60),      # left lower lip corner
    8: (0, 180),      # left upper lip
    9: (10, 40),      # left upper lip corner
    10: (55, 100),    # jaw mirrored (zmirrorowany zakres 1: 180-125 i 180-80)
    11: (10, 111),    # right eye lower eyelid
    12: (30, 115),    # right eye upper eyelid
    13: (40, 120),    # eyes up-down
    14: (50, 150),    # eyes right-left
    15: (20, 90),     # left eye upper eyelid
    16: (25, 90),     # left eye lower eyelid
    17: (70, 150),    # right eyebrow outer
    18: (70, 160),    # right eyebrow inner
    19: (50, 140),    # left eyebrow inner
    20: (20, 100)     # left eyebrow outer
}

last_blink_time = 0
blink_interval = 2.0
is_blinking = False
blink_start_time = 0

def set_servo_angle(s_id, target_angle):
    if s_id < 0 or s_id > 20:
        print("Nieprawidłowe ID serwa")
        return

    min_ang, max_ang = limits[s_id]
    safe_angle = max(min_ang, min(max_ang, target_angle))

    servos[s_id].angle = safe_angle
    print(f"Serwo {s_id} -> ustawione na: {safe_angle} stopni")

    # lustrzane odbcie szczęki
    if s_id in (1, 10):
        other_id = 10 if s_id == 1 else 1
        other_min, other_max = limits[other_id]
        
        mirror_angle = 180 - safe_angle
        
        mirror_angle = max(other_min, min(other_max, mirror_angle))
        
        servos[other_id].angle = mirror_angle
        print(f"Lustrzane serwo {other_id} -> ustawione na: {mirror_angle} stopni")

def move_eye_servo(s_id, target_angle):
    min_ang, max_ang = limits[s_id]
    safe_angle = max(min_ang + 5, min(max_ang - 5, target_angle))
    servos[s_id].angle = safe_angle

def handle_random_blink():
    global last_blink_time, blink_interval, is_blinking, blink_start_time
    current_time = time.time()
    
    # Zamknięcie oczu
    if not is_blinking and (current_time - last_blink_time) > blink_interval:
        is_blinking = True
        blink_start_time = current_time
        
        move_eye_servo(12, 35)
        move_eye_servo(11, 106) # Prawe
        move_eye_servo(15, 85)
        move_eye_servo(16, 30)  # Lewe

    # Otwarcie oczu po 150ms
    if is_blinking and (current_time - blink_start_time) > 0.150:
        move_eye_servo(12, 110)
        move_eye_servo(11, 15)  # Prawe
        move_eye_servo(15, 25)
        move_eye_servo(16, 85)  # Lewe
        
        is_blinking = False
        last_blink_time = current_time
        blink_interval = random.uniform(1.5, 5.0)

def delay_and_blink(ms):
    end_time = time.time() + (ms / 1000.0)
    while time.time() < end_time:
        handle_random_blink()
        time.sleep(0.01)

def animate_eyes_complex():
    print("Start animacji: Skosy i mruganie... (Użyj Ctrl+C aby wrócić do menu)")
    try:
        while True:
            move_eye_servo(14, 145); delay_and_blink(400)
            move_eye_servo(14, 55);  delay_and_blink(400)
            
            move_eye_servo(14, 55); move_eye_servo(13, 115); delay_and_blink(600)
            move_eye_servo(14, 145); move_eye_servo(13, 45); delay_and_blink(600)
            move_eye_servo(14, 145); move_eye_servo(13, 115); delay_and_blink(600)
            move_eye_servo(14, 55); move_eye_servo(13, 45); delay_and_blink(600)

            move_eye_servo(14, 100)
            move_eye_servo(13, 60)
            delay_and_blink(50) 
    except KeyboardInterrupt:
        print("\nPrzerwano animację. Powrót do wprowadzania ręcznego.")

def animate_eyes():
    print("Start animacji oczu... (Użyj Ctrl+C aby wrócić do menu)")
    try:
        while True:
            move_eye_servo(14, 145); delay_and_blink(500) # Lewo
            move_eye_servo(14, 55);  delay_and_blink(500) # Prawo
            move_eye_servo(14, 100); delay_and_blink(300) # Środek

            move_eye_servo(13, 115); delay_and_blink(500) # Góra
            move_eye_servo(13, 45);  delay_and_blink(500) # Dół
            move_eye_servo(13, 60);  delay_and_blink(300) # Środek
            
            # Zamknięcie
            move_eye_servo(12, 35); move_eye_servo(11, 106)
            move_eye_servo(15, 85); move_eye_servo(16, 30)
            delay_and_blink(200)

            # Otwarcie
            move_eye_servo(12, 110); move_eye_servo(11, 15)
            move_eye_servo(15, 25);  move_eye_servo(16, 85)
            delay_and_blink(300)
            
            # Oczy szeroko otwarte, lekko w górę
            move_eye_servo(13, 80)
            move_eye_servo(12, 115) 
            move_eye_servo(15, 20)
            delay_and_blink(1000)
            
            # Powrót do neutralnego
            move_eye_servo(13, 60)
            move_eye_servo(14, 100)
            delay_and_blink(200)
    except KeyboardInterrupt:
        print("\nPrzerwano animację. Powrót do wprowadzania ręcznego.")

if __name__ == "__main__":
    print("\nInicjalizacja serwomechanizmów na pozycje środkowe...")
    for s_id in range(len(servos)):
        if s_id in limits:
            min_ang, max_ang = limits[s_id]
            mid_angle = (min_ang + max_ang) / 2
            set_servo_angle(s_id, mid_angle)
        time.sleep(0.02)

    print("\nSystem gotowy. Wpisz komendę w formacie: [ID] [KĄT] (np., '1 90').")
    print("Opcje specjalne:")
    print("  98 - Uruchamia standardową animację oczu")
    print("  99 - Uruchamia zaawansowaną animację oczu (ze skosami)")
    print("  Ctrl+C - Całkowite wyjście z programu")
    
    try:
        while True:
            cmd = input("> ")
            parts = cmd.split()
            if len(parts) == 1:
                try:
                    s_id = int(parts[0])
                    if s_id == 99:
                        animate_eyes_complex()
                    elif s_id == 98:
                        animate_eyes()
                    else:
                        print("Wymagany format: [ID] [KĄT]. Komendy pojedyncze to tylko 98 lub 99.")
                except ValueError:
                    print("Proszę wpisać prawidłowe liczby.")
                    
            elif len(parts) == 2:
                try:
                    s_id = int(parts[0])
                    angle = float(parts[1])
                    set_servo_angle(s_id, angle)
                except ValueError:
                    print("Proszę wpisać prawidłowe liczby.")
    except KeyboardInterrupt:
        print("\nZamykanie programu. Wyłączanie sygnałów PWM.")
        pca0.deinit()
        pca1.deinit()