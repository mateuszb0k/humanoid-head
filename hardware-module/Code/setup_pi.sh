#!/bin/bash

sudo apt update 
sudo apt upgrade -y

sudo apt install -y python3-pip python3-dev i2c-tools

sudo raspi-config nonint do_i2c 0

pip3 install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor --break-system-packages

i2cdetect -y 1