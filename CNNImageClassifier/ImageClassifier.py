import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
