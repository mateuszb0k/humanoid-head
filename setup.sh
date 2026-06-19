#!/bin/bash
set -e

echo "Starting setup"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This script works on Debian/Ubuntu/Raspberry Pi OS/Jetson Ubuntu systems with apt-get."
    exit 1
fi

sudo apt-get update

sudo apt-get install -y \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    i2c-tools \
    libcamera-dev \
    python3-libcamera \
    python3-kms++ \
    libgl1-mesa-glx \
    libgl1 \
    libglib2.0-0 \
    libhdf5-dev \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg \
    wget

echo "Checking models"

if [ ! -f "face_detection_yunet_2023mar.onnx" ]; then
    echo "Downloading detection model"
    wget -q --show-progress https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
fi

if [ ! -f "face_recognition_sface_2021dec.onnx" ]; then
    echo "Downloading recognition model"
    wget -q --show-progress https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
fi

echo "Checking uv"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is still not available."
    echo "Close and reopen the terminal, or add ~/.local/bin to PATH."
    exit 1
fi

echo "Installing Python 3.12 with uv"
uv python install 3.12

echo "Setting up virtual environment"

if [ ! -d "venv" ]; then
    python3 -m venv --system-site-packages venv
fi

echo "Installing packages in virtual environment"

source venv/bin/activate

pip install --upgrade pip

pip install \
    opencv-python \
    numpy \
    Flask \
    requests \
    tensorflow

pip install \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-motor \
    adafruit-extended-bus

deactivate

if [ -d "speech-module" ]; then
    echo "Installing speech-module dependencies"
    cd speech-module
    uv sync
    cd ..
else
    echo "Skipping speech-module - directory not found"
fi

if [ -d "vision-module" ]; then
    echo "Installing vision-module dependencies"
    cd vision-module
    uv sync
    cd ..
else
    echo "Skipping vision-module - directory not found"
fi

echo "Checking Ollama"

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found, installing"
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed"
fi

echo ""
echo "Done"
echo ""
echo "Examples:"
echo "  source venv/bin/activate"
echo "  cd speech-module && uv run python nlp_pipeline.py"
echo "  cd speech-module && uv run python nlp_pipeline_jetson_test.py"
echo "  cd vision-module && uv run python <file.py>"
