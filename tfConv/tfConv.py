#####################################################################################
# kerasConv.py
#
# Dev. Dongwon Paek
# Description: Real example of converting keras weight file to .tflite file
#####################################################################################

import tensorflow as tf

export_dir = './epoch-50.data-00001-of-00002'

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)

print("Model successfully converted into tflite file.")