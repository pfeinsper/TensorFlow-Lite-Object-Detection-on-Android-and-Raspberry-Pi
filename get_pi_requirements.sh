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
#cd content
#wget https://github.com/pedromtelho/VMobi-ocr-dependencies/raw/master/east_model_float16.tflite
#cd opencv_text_detection
#wget https://github.com/pedromtelho/VMobi-ocr-dependencies/raw/master/frozen_east_text_detection.pb
#sudo apt install tesseract-ocr
#pip3 install pytesseract
#pip3 install autocorrect

# Get packages required for TensorFlow
# Using the tflite_runtime packages available at https://www.tensorflow.org/lite/guide/python
# Will change to just 'pip3 install tensorflow' once newer versions of TF are added to piwheels

#pip3 install tensorflow

#version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

#if [ $version == "3.7" ]; then
#pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
#fi

#if [ $version == "3.5" ]; then
#pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp35-cp35m-linux_armv7l.whl
#fi

