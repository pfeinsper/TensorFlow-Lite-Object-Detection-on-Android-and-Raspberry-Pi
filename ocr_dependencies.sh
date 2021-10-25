#!/bin/bash
cd content
wget https://github.com/pedromtelho/VMobi-ocr-dependencies/raw/master/east_model_float16.tflite
cd opencv_text_detection
wget https://github.com/pedromtelho/VMobi-ocr-dependencies/raw/master/frozen_east_text_detection.pb
sudo apt install tesseract-ocr
