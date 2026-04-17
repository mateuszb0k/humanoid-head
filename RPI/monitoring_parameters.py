# monitoring CPU, temperature, and processes in real-time
btop

# checking official fan speed thresholds and parameters
dtoverlay -h rpi-poe

# reading the current CPU temperature - one-time check
vcgencmd measure_temp

# checking current fan state (0 = OFF, 1-4 = Increasing Speed)
cat /sys/class/thermal/cooling_device0/cur_state