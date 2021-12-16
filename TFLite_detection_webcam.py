######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os, argparse, cv2
import numpy as np
import sys, time, queue
from threading import Thread
import importlib.util

# Audio Setup
from play_voice import play_voice

# GPIO - Pi Buttons
from gpiozero import Button

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	    # Return the most recent frame
        return self.frame

    def stop(self):
	    # Indicate that the camera and thread should be stopped
        print("Stopping videostream")
        self.stopped = True

def safari_mode(args, query_button, fila, lang="en", ptbr_categ=None):
    """Runs the Safari Mode; args is a tuple"""
    interpreter, imW, imH, width, height, floating_model, input_mean, input_std, input_details, output_details, min_conf_threshold, labels = args
    timeout_categs = {"???": 0, "person": 0, "bicycle": 0, "car": 0, "motorcycle": 0, "airplane": 0, "bus": 0, "train": 0, "truck": 0, "boat": 0, "traffic light": 0, "fire hydrant": 0, "stop sign": 0, "parking meter": 0, "bench": 0, "bird": 0, "cat": 0, "dog": 0, "horse": 0, "sheep": 0, "cow": 0, "elephant": 0, "bear": 0, "zebra": 0, "giraffe": 0, "backpack": 0, "umbrella": 0, "handbag:": 0, "tie": 0, "suitcase": 0, "frisbee": 0, "skis": 0, "snowboard": 0, "sports ball": 0, "kite": 0, "baseball bat": 0, "baseball glove": 0, "skateboard": 0, "surfboard": 0, "tennis racket": 0, "bottle": 0, "wine glass": 0, "cup": 0, "fork": 0, "knife": 0, "spoon": 0, "bowl": 0, "banana": 0, "apple": 0, "sandwich": 0, "orange": 0, "broccoli": 0, "carrot": 0, "hot dog": 0, "pizza": 0, "donut": 0, "cake": 0, "chair": 0, "couch": 0, "potted plant": 0, "bed": 0, "dining table": 0, "toilet": 0, "tv:": 0, "laptop": 0, "mouse": 0, "remote": 0, "keyboard": 0, "cell phone": 0, "microwave": 0, "oven": 0, "toaster": 0, "sink": 0, "refrigerator": 0, "book": 0, "clock": 0, "vase": 0, "scissors": 0, "teddy bear": 0, "hair drier": 0, "toothbrush": 0}
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)
    if (lang=="pt-br"):
        # play_voice("Modo safari foi ativado", lang=lang[:2])
        fila.put("Modo safari foi ativado")
    else:
        # play_voice("Safari mode is activated")
        fila.put("Safari mode is activated")
    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    out = 0
    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if (scores[i] > 0.8):
                    if (time.time() - timeout_categs[object_name] >= 3):
                        if ((xmin + xmax)/2 > 2*imW/3):
                            if (lang == "pt-br"):
                                # play_voice(f"{ptbr_categ[object_name]} à sua direita", lang[:2])
                                fila.put(f"{ptbr_categ[object_name]} à sua direita")
                            else:
                                # play_voice(f"{object_name} at your right")
                                fila.put(f"{object_name} at your right")
                        elif ((xmin + xmax)/2 < imW/3):
                            if (lang == "pt-br"):
                                # play_voice(f"{ptbr_categ[object_name]} à sua esquerda", lang[:2])
                                fila.put(f"{ptbr_categ[object_name]} à sua esquerda")
                            else:
                                # play_voice(f"{object_name} at your left")
                                fila.put(f"{object_name} at your left")
                        else:
                            if (lang == "pt-br"):
                                # play_voice(f"{ptbr_categ[object_name]} à sua frente", lang[:2])
                                fila.put(f"{ptbr_categ[object_name]} à sua frente")
                            else:
                                # play_voice(f"{object_name} in front of you")
                                fila.put(f"{object_name} in front of you")
                        timeout_categs[object_name] = time.time()

                # Draw label
                
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Safari Mode', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        if query_button.is_pressed:
            out = 1
            break

    # Clean up  
    cv2.destroyAllWindows()
    videostream.stop()
    return out
    # if t.do_run:
    #     t.do_run = False

def query_mode(args, query_obj, query_btn, fila, lang="en", ptbr_categ=None):
    """Runs the query mode"""
    interpreter, imW, imH, width, height, floating_model, input_mean, input_std, input_details, output_details, min_conf_threshold, labels = args
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    # counter = 0
    breakFlag = False
    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                if (object_name == query_obj):
                    # if (counter >= 3):
                    if ((xmin + xmax)/2 > 2*imW/3):
                        if (lang == "pt-br"):
                            # play_voice(f"Achei o objeto {ptbr_categ[query_obj]}! Está à sua direita.", lang[:2])
                            fila.put(f"Achei o objeto {ptbr_categ[query_obj]}! Está à sua direita.")
                        else:
                            # play_voice(f"Found the {query_obj}! It is at your right.")
                            fila.put(f"Found the {query_obj}! It is at your right.")
                    elif ((xmin + xmax)/2 < imW/3):
                        if (lang == "pt-br"):
                            # play_voice(f"Achei o objeto {ptbr_categ[query_obj]}! Está à sua esquerda.", lang[:2])
                            fila.put(f"Achei o objeto {ptbr_categ[query_obj]}! Está à sua esquerda.")
                        else:
                            # play_voice(f"Found the {query_obj}! It is at your left.")
                            fila.put(f"Found the {query_obj}! It is at your left.")
                    else:
                        if (lang == "pt-br"):
                            # play_voice(f"Achei o objeto {ptbr_categ[query_obj]}! Está à sua frente.", lang[:2])
                            fila.put(f"Achei o objeto {ptbr_categ[query_obj]}! Está à sua frente.")
                        else:
                            # play_voice(f"Found the {query_obj}! It is in front of you.")
                            fila.put(f"Found the {query_obj}! It is in front of you.")
                    breakFlag = True
                    break

                    # counter += 1
                # else:
                #     counter = 0
        
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Query Mode', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q') or breakFlag or query_btn.is_held:
            if (lang[:2]=="pt"):
                # play_voice("Retornando ao Modo Safari")
                fila.put("Retornando ao Modo Safari")
            else:
                # play_voice("Returning to Safari Mode")
                fila.put("Returning to Safari Mode")
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()

def initialize_detector(args):
    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate
    
    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'      
    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5
    args = (interpreter, imW, imH, width, height, floating_model, input_mean, input_std, input_details, output_details, min_conf_threshold, labels)
    return args