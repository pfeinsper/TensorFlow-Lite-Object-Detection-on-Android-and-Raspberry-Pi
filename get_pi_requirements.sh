#!/bin/bash

# Get packages required for OpenCV

sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

pip install --upgrade pip
# Need to get an older version of OpenCV because version 4 has errors
pip3 install -r requirements.txt

# Audio dependencies
sudo apt-get install -y pavucontrol
sudo apt-get install -y libatlas-base-dev libportaudio0 libportaudio2 libportaudiocpp0 	portaudio19-dev
sudo apt-get install -y flac

# tflite installation
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl

# OCR dependencies
cd text_recognition
wget https://github.com/pedromtelho/VMobi-ocr-dependencies/raw/master/east_model_float16.tflite
sudo apt install -y tesseract-ocr
pip3 install pytesseract
pip3 install autocorrect


