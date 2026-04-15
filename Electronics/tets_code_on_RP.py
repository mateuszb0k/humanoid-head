import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

"""
Requirements before the test:

In the terminal, check if I2C is enabled:
    $ sudo raspi-config

Install the required libraries:
    $ pip3 install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor
"""

# I2C Configuration
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize PCA9685 drivers
pca0 = PCA9685(i2c, address=0x40)
pca0.frequency = 50
pca1 = PCA9685(i2c, address=0x41)
pca1.frequency = 50

# Conversion to microseconds (for Python library):
# 150 / 4096 * 20000us = ~732 us
# 600 / 4096 * 20000us = ~2930 us
MIN_PULSE = 732
MAX_PULSE = 2930

# Pin mapping: first 16 to pca0, next 5 to pca1
servos = []
for i in range(11):
    servos.append(servo.Servo(pca0.channels[i], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE))
for i in range(10):
    servos.append(servo.Servo(pca1.channels[i], min_pulse=MIN_PULSE, max_pulse=MAX_PULSE))

limits = {
    0: (0, 180),      # tongue
    1: (80, 125),     # jaw
    2: (50, 65),      # right upper lip corner
    3: (20, 80),      # right upper lip
    4: (50, 70),      # right lower lip corner
    5: (0, 180),      # right lower lip (moves weirdly - keeping full range for testing)
    6: (45, 110),     # left lower lip
    7: (40, 60),      # left lower lip corner
    8: (0, 180),      # left upper lip (moves weirdly)
    9: (10, 40),      # left upper lip corner
    10: (80, 125),    # jaw mirrored (mapped to servo 1 range)
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

def set_servo_angle(s_id, target_angle):
    if s_id < 0 or s_id > 20:
        print("Invalid servo ID")
        return

    # Constrain angle to defined limits
    min_ang, max_ang = limits[s_id]
    safe_angle = max(min_ang, min(max_ang, target_angle))

    # Set the angle
    servos[s_id].angle = safe_angle
    print(f"ID: {s_id} -> {safe_angle} degrees")

# Main terminal control loop
if __name__ == "__main__":
    print("Enter command in format: [ID] [ANGLE] (e.g., '1 100'). Ctrl+C to exit.")
    try:
        while True:
            cmd = input("> ")
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    s_id = int(parts[0])
                    angle = float(parts[1])
                    set_servo_angle(s_id, angle)
                except ValueError:
                    print("Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\nExiting.")
        # Disable PWM signals
        pca0.deinit()
        pca1.deinit()