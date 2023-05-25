import waggle.plugin as plugin
from random import random
from time import sleep

plugin.init()

import PIL
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import resnet50
#import keras2onnx
#import onnxruntime
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model

import tensorflow as tf
from tensorflow import keras
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def to_str(var):
    if type(var) is list:
        return str(var)[1:-1] # list
    if type(var) is np.ndarray:
        try:
            return str(list(var[0]))[1:-1] # numpy 1D array
        except TypeError:
            return str(list(var))[1:-1] # numpy sequence
    return str(var) # everything else

model= tf.keras.applications.VGG19( include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax", )
tf.saved_model.save(model,'temp_model1')

while True:

    for file in os.listdir('./test_images'):

        filename = os.path.join('./test_images',file)
        original = load_img(filename, target_size = (224, 224))


        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis = 0)

        processed_image = resnet50.preprocess_input(image_batch.copy())

        predictions = model.predict(processed_image)
        label = decode_predictions(predictions)
        result=filename+'   '+to_str(label[0][0])
        print('publishing label',result)
        plugin.publish('env.classification', result)
        sleep(1)
