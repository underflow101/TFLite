##########################################################################
# tfliteInference.py
# Dev. Dongwon Paek
#
# Description: This source code infer camera's ongoing video with
#              TFLite.
##########################################################################

import time
from picamera import Picamera
import cv2

with Picamera() as cam:
    cam.start_preview(fullscreen=False, window=(100,20,640,480))
    