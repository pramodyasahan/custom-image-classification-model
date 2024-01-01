import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = tf.keras.utils.image_dataset_from_directory('data')
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()

train_size = int(len(data) * 0.7) + 2
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)
print(len(data))
print(train_size + val_size + test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
