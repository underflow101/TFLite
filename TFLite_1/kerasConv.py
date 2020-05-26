####################################################################################
# kerasConv.py
#
# Dev. Dongwon Paek
# Description: Real example of converting keras weight file to .tflite file
####################################################################################

import tensorflow as tf
import h5py
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model
from model import SqueezeNet


model = SqueezeNet(input_shape = (224, 224, 3), nb_classes=4)
model.load_weights('lpl.h5')
model.summary()
print("Weight loaded complete.")

sgd = SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled.")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)

print("Model successfully converted into tflite file.")
