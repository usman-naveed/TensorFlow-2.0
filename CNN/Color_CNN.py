import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib as plt
import numpy as np
import logging

logger = tf.get_logger
logger.setLevel(logging.ERROR)

#downloading the data from external URL, using tf.get_file()
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin= _URL, extract=True)

#set up directory paths
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
training_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats = os.path.join(training_dir, 'cats')
train_dogs = os.path.join(training_dir, 'dogs')
validation_cats = os.path.join(validation_dir, 'cats')
validation_dogs = os.path.join(validation_dir, 'dogs')
