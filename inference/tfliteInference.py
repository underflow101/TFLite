##########################################################################
# tfliteInference.py
# Dev. Dongwon Paek
#
# Description: This source code infer camera's ongoing video with
#              TFLite.
##########################################################################

import time, timeit
from picamera import Picamera
import cv2
from imutils.video import VideoStream
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import tensorflow as tf


def checkFPS(prevTime):
    curTime = time.time()
    period = curTime - prevTime
    prevTime = curTime
    fps = 1 / period
    return prevTime, fps

# load tflite model
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


cap = cv2.VideoCapture(0)
while cap.isOpened():
	success, frame = cap.read()
	if success:
		cv2.imshow("test", frame)
        prevTime, fps = checkFPS(prevTime)
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 30, 20), 2)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
			break

cap.release()
cv2.destroyAllWindows()