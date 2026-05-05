import time
import random
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# Inicjalizacja magistrali I2C i sterowników
i2c = busio.I2C(board.SCL, board.SDA)
driverSO = PCA9685(i2c, address=0x40)
driverR = PCA9685(i2c, address=0x41)

driverSO.frequency = 50
driverR.frequency = 50

# Zmienne dla mrugania
last_blink_time = 0
blink_interval = 2.0
is_blinking = False
blink_start_time = 0

# Konfiguracja serw (id: {sterownik, pin, min, max, wlaczone})
# sterownik: 0 = driverSO, 1 = driverR
servos_config = {
    0:  {'driver': 0, 'pin': 0,  'min': 0,  'max': 180, 'wlaczone': False}, # jezyk
    1:  {'driver': 0, 'pin': 1,  'min': 80, 'max': 125, 'wlaczone': False},  # szczeka max otwarta 80, zamknieta 125
    2:  {'driver': 0, 'pin': 2,  'min': 50, 'max': 65,  'wlaczone': True},  # prawy korner gorne
    3:  {'driver': 0, 'pin': 3,  'min': 20, 'max': 80,  'wlaczone': True},  # prawa warga gorna
    4:  {'driver': 0, 'pin': 4,  'min': 50, 'max': 70,  'wlaczone': True},  # prawy korner dolne
    5:  {'driver': 0, 'pin': 5,  'min': 25, 'max': 92,  'wlaczone': True},  # prawa warga dolna
    6:  {'driver': 0, 'pin': 6,  'min': 45, 'max': 110, 'wlaczone': True},  # lewa warga dolna
    7:  {'driver': 0, 'pin': 7,  'min': 40, 'max': 60,  'wlaczone': True},  # lewy korner dolne
    8:  {'driver': 0, 'pin': 8,  'min': 35, 'max': 85,  'wlaczone': True},  # lewa gorna warga
    9:  {'driver': 0, 'pin': 9,  'min': 10, 'max': 40,  'wlaczone': True},  # lewy korner gorne
    10: {'driver': 0, 'pin': 10, 'min': 45, 'max': 80,  'wlaczone': True},  # szczeka niezjarana
    
    11: {'driver': 1, 'pin': 0,  'min': 0,  'max': 80,  'wlaczone': True},  # prawe oko dolna powieka
    12: {'driver': 1, 'pin': 1,  'min': 30, 'max': 115, 'wlaczone': True},  # gorna powieka
    13: {'driver': 1, 'pin': 2,  'min': 40, 'max': 120, 'wlaczone': True},  # oczy gora-dol
    14: {'driver': 1, 'pin': 3,  'min': 50, 'max': 150, 'wlaczone': True},  # oczy prawo-lewo
    15: {'driver': 1, 'pin': 4,  'min': 20, 'max': 90,  'wlaczone': True},  # lewe oko gorna powieka
    16: {'driver': 1, 'pin': 5,  'min': 25, 'max': 90,  'wlaczone': True},  # lewe oko dolna powieka
    17: {'driver': 1, 'pin': 6,  'min': 70, 'max': 150, 'wlaczone': True},  # prawa brew zewnatrz
    18: {'driver': 1, 'pin': 7,  'min': 70, 'max': 160, 'wlaczone': True},  # prawa brew srodek
    19: {'driver': 1, 'pin': 8,  'min': 80, 'max': 170, 'wlaczone': True},  # lewa brew srodek
    20: {'driver': 1, 'pin': 9,  'min': 20, 'max': 100, 'wlaczone': True}   # lewa brew zewnatrz
}

# Utworzenie obiektów serw
servo_objects = {}
for servo_id, cfg in servos_config.items():
    drv = driverSO if cfg['driver'] == 0 else driverR
    servo_objects[servo_id] = servo.Servo(drv.channels[cfg['pin']], min_pulse=730, max_pulse=2930)

def set_servo_angle(num, angle, use_margin=False):
    """Główna funkcja ustawiająca kąt serwa z uwzględnieniem limitów i zabezpieczeń."""
    cfg = servos_config.get(num)
    if not cfg or not cfg['wlaczone']:
        return
    
    margin = 5 if use_margin else 0
    # Ustalenie faktycznego minimum i maksimum (zabezpieczenie gdy min > max w słowniku)
    limit_min = min(cfg['min'], cfg['max']) + margin
    limit_max = max(cfg['min'], cfg['max']) - margin
    
    # Ograniczenie kąta do dozwolonego zakresu
    safe_angle = max(limit_min, min(angle, limit_max))
    
    # Ustawienie kąta sprzętowo
    servo_objects[num].angle = safe_angle

def move_eye_servo(num, angle):
    """Funkcja pomocnicza zachowana dla animacji z marginesem 5 stopni."""
    set_servo_angle(num, angle, use_margin=True)

def handle_random_blink():
    global last_blink_time, is_blinking, blink_start_time, blink_interval
    current_time = time.time()

    if not is_blinking and (current_time - last_blink_time) > blink_interval:
        is_blinking = True
        blink_start_time = current_time
        
        move_eye_servo(12, 35)
        move_eye_servo(11, 75)
        move_eye_servo(15, 85)
        move_eye_servo(16, 30)

    if is_blinking and (current_time - blink_start_time) > 0.150:
        move_eye_servo(12, 110)
        move_eye_servo(11, 0)
        move_eye_servo(15, 25)
        move_eye_servo(16, 85)
        
        is_blinking = False
        last_blink_time = current_time
        blink_interval = random.uniform(1.5, 5.0)

def animate_eyes_complex():
    print("Start animacji: Skosy i mruganie... (Ctrl+C aby przerwać)")
    try:
        while True:
            handle_random_blink()
            move_eye_servo(14, 145); time.sleep(0.4)
            move_eye_servo(14, 55);  time.sleep(0.4)
            
            move_eye_servo(14, 55); move_eye_servo(13, 115); time.sleep(0.6)
            move_eye_servo(14, 145); move_eye_servo(13, 45); time.sleep(0.6)
            move_eye_servo(14, 145); move_eye_servo(13, 115); time.sleep(0.6)
            move_eye_servo(14, 55); move_eye_servo(13, 45); time.sleep(0.6)
            
            move_eye_servo(14, 100); move_eye_servo(13, 60)
    except KeyboardInterrupt:
        print("\nAnimacja zatrzymana.")

def animate_radosc():
    print("Start animacji radosc")
    moves = [
        [(14, 100), (13, 60)],
        [(11, 50), (12, 95), (15, 40), (16, 55)],
        [(10, 45)],
        [(19, 110), (18, 130)],
        [(20, 50), (17, 110)],
        [(3, 65), (8, 35), (5, 50), (6, 75)],
        [(2, 100), (4, 30), (7, 110), (9, 10)]
    ]
    for step in moves:
        for servo_id, angle in step:
            move_eye_servo(servo_id, angle)
        time.sleep(0.04)
    print("Koniec animacji radosc")

def animate_smutek():
    print("Start animacji smutek")
    moves = [
        [(14, 100), (13, 60)],
        [(11, 25), (12, 95), (15, 40), (16, 70)],
        [(10, 45)],
        [(19, 110), (18, 130)],
        [(20, 100), (17, 70)],
        [(3, 20), (8, 85), (5, 40), (6, 100)],
        [(2, 40), (4, 50), (7, 30), (9, 60)]
    ]
    for step in moves:
        for servo_id, angle in step:
            move_eye_servo(servo_id, angle)
        time.sleep(0.04)
    print("Koniec animacji smutek")

def animate_szok():
    print("Start animacji szok")
    moves = [
        [(14, 100), (13, 60)],
        [(11, 0), (12, 115), (15, 20), (16, 90)],
        [(10, 80)],
        [(19, 170), (18, 70)],
        [(20, 20), (17, 150)],
        [(3, 20), (8, 85), (5, 90), (6, 45)],
        [(2, 65), (4, 60), (7, 50), (9, 40)]
    ]
    for step in moves:
        for servo_id, angle in step:
            move_eye_servo(servo_id, angle)
        time.sleep(0.04)
    print("Koniec animacji szok")

def animate_neutral():
    print("Start animacji neutral")
    moves = [
        [(14, 100), (13, 60)],
        [(11, 25), (12, 95), (15, 40), (16, 70)],
        [(10, 45)],
        [(19, 110), (18, 130)],
        [(20, 50), (17, 110)],
        [(3, 20), (8, 85), (5, 90), (6, 45)],
        [(2, 65), (4, 60), (7, 50), (9, 40)]
    ]
    for step in moves:
        for servo_id, angle in step:
            move_eye_servo(servo_id, angle)
        time.sleep(0.04)
    print("Koniec animacji neutral")

def animate_eyes():
    print("Start animacji oczu... (Ctrl+C aby przerwać)")
    try:
        while True:
            handle_random_blink()
            move_eye_servo(14, 145); time.sleep(0.5)
            move_eye_servo(14, 55);  time.sleep(0.5)
            move_eye_servo(14, 100); time.sleep(0.3)

            move_eye_servo(13, 115); time.sleep(0.5)
            move_eye_servo(13, 45);  time.sleep(0.5)
            move_eye_servo(13, 60);  time.sleep(0.3)

            move_eye_servo(12, 35); move_eye_servo(11, 75)
            move_eye_servo(15, 85); move_eye_servo(16, 30)
            time.sleep(0.2)

            move_eye_servo(12, 110); move_eye_servo(11, 0)
            move_eye_servo(15, 25);  move_eye_servo(16, 85)
            time.sleep(0.3)

            move_eye_servo(13, 80); move_eye_servo(12, 115); move_eye_servo(15, 20)
            time.sleep(1.0)

            move_eye_servo(13, 60); move_eye_servo(14, 100)
    except KeyboardInterrupt:
        print("\nAnimacja zakończona.")

def main():
    print("System gotowy.")
    print("Wpisz komendę w formacie: [nr_serwa] [kat] (np. '5 90').")
    print("Komendy animacji: 94 (Radosc), 95 (Smutek), 96 (Neutral), 97 (Szok), 98 (Oczy), 99 (Oczy zxlozone).")
    
    while True:
        try:
            cmd = input("Komenda: ").strip().split()
            if len(cmd) < 1:
                continue
            
            servo_num = int(cmd[0])
            
            if servo_num == 99:
                animate_eyes_complex()
            elif servo_num == 98:
                animate_eyes()
            elif servo_num == 97:
                animate_szok()
            elif servo_num == 96:
                animate_neutral()
            elif servo_num == 95:
                animate_smutek()
            elif servo_num == 94:
                animate_radosc()
            else:
                if len(cmd) >= 2:
                    angle = int(cmd[1])
                    if 0 <= servo_num < 21:
                        set_servo_angle(servo_num, angle)
                        print(f"Serwo {servo_num} ustawione na (sugerowany): {angle}")
                        
                        # Logika lustrzana dla szczęki
                        if servo_num in (1, 10):
                            other_servo = 10 if servo_num == 1 else 1
                            cfg = servos_config[other_servo]
                            mirror_angle = cfg['min'] + cfg['max'] - angle
                            set_servo_angle(other_servo, mirror_angle)
                            print(f"Lustrzane serwo {other_servo} ustawione na: {mirror_angle}")
                    else:
                        print("Bledny numer serwa! (Zakres 0-20)")
        except ValueError:
            print("Błędny format danych.")
        except KeyboardInterrupt:
            print("\nZamykanie systemu...")
            break

if __name__ == "__main__":
    main()