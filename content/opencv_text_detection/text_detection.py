# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import pytesseract
from autocorrect import Speller
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

from TFLite_detection_webcam import play_voice

spell = Speller(fast=True)

def perform_inference(tflite_path, preprocessed_image):
    interpreter = Interpreter(model_path=tflite_path,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    input_details = interpreter.get_input_details()

    if input_details[0]["dtype"] == np.uint8:
        print("Integer quantization!")
        input_scale, input_zero_point = input_details[0]["quantization"]
        preprocessed_image = preprocessed_image / input_scale + input_zero_point
    preprocessed_image = preprocessed_image.astype(input_details[0]["dtype"])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

    start = time.time()
    interpreter.invoke()
    print(f"Inference took: {time.time()-start} seconds")

    scores = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])()
    geometry = interpreter.tensor(
        interpreter.get_output_details()[1]['index'])()

    return scores, geometry


def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.6:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)



def main_text_detection():
    # initialize the original frame dimensions, new frame dimensions,
    # and ratio between the dimensions
    (W, H) = (None, None)
    (newW, newH) = (320, 320)
    (rW, rH) = (None, None)


    print("[INFO] starting video stream...")
    # define camera source number
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # start the FPS throughput estimator
    fps = FPS().start()
    
    t0 = time.time()
    # loop over frames from the video stream
    while True:
        if (time.time() - t0) <= 2:
            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            frame = vs.read()

            # check to see if we have reached the end of the stream
            if frame is None:
                break

            # resize the frame, maintaining the aspect ratio
            frame = imutils.resize(frame, width=1000)
            orig = frame.copy()

            # if our frame dimensions are None, we still need to compute the
            # ratio of old frame dimensions to new frame dimensions
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                rW = W / float(newW)
                rH = H / float(newH)

            # resize the frame, this time ignoring aspect ratio
            frame = cv2.resize(frame, (newW, newH))

            (H, W) = frame.shape[:2]

            # convert the frame to a floating point data type and perform mean
            # subtraction
            frame = frame.astype("float32")
            mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
            frame -= mean
            frame = np.expand_dims(frame, 0)

            scores, geometry = perform_inference(tflite_path="/home/pi/VMobi-objetc-detection-raspberry-pi/content/east_model_float16.tflite",
                                                 preprocessed_image=frame)

            scores = np.transpose(scores, (0, 3, 1, 2))
            geometry = np.transpose(geometry, (0, 3, 1, 2))

            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)

            results = []
            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)

                # # in order to obtain a better OCR of the text we can potentially
                # # apply a bit of padding surrounding the bounding box -- here we
                # # are computing the deltas in both the x and y directions
                # dX = int((endX - startX) * -0.05)  # padding
                # dY = int((endY - startY) * 0.14)  # padding
                # # apply padding to each side of the bounding box, respectively
                # startX = max(0, startX - dX)
                # startY = max(0, startY - dY)
                # endX = min(origW, endX + (dX * 2))
                # endY = min(origH, endY + (dY * 2))
                # extract the actual padded ROI
                roi = orig[startY:endY, startX:endX]
                try:
                    # in order to apply Tesseract v4 to OCR text we must supply
                    # (1) a language, (2) an OEM flag of 4, indicating that the we
                    # wish to use the LSTM neural net model for OCR, and finally
                    # (3) an OEM value, in this case, 7 which implies that we are
                    # treating the ROI as a single line of text
                    config = ("-l eng --oem 1 --psm 7")
                    text = pytesseract.image_to_string(roi, config=config)

                    print("OCR TEXT VERIFICATION")

                    text = text.lower()
                    
                    text = spell(text)
                    play_voice(text)
                    # add the bounding box coordinates and OCR'd text to the list
                    # of results
                    results.append(((startX, startY, endX, endY), text))
                    # sort the results bounding box coordinates from top to bottom

                    results = sorted(results, key=lambda r: r[0][1])
                    # loop over the results
                    for ((startX, startY, endX, endY), text) in results:

                        # draw the bounding box on the frame
                        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
                except Exception as e:
                    print(e)
            else:
                t0 = time.time()
        # update the FPS counter
        fps.update()

        # show the output frame
        cv2.imshow("Text Detection", orig)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # if we are using a webcam, release the pointer
    vs.stop()



    # close all windows
    cv2.destroyAllWindows()