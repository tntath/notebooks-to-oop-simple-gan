import logging
import random
import numpy as np
import tensorflow as tf
from keras.datasets import mnist

from gan import Gan

# random.seed(42)  # Python's built-in random module
# np.random.seed(42)  # NumPy
# tf.random.set_seed(42)  # TensorFlow


# Load the dataset
(X_train, _), (_, _) = mnist.load_data()

# Normalize the images -convert pixel values from [0,255] to [-1, 1]
X_train = X_train / 127.5 - 1

X_train = np.expand_dims(X_train, axis=3)

print(X_train.shape)

gan = Gan()
print(gan.discriminator.summary())
print(gan.generator.summary())
print(gan.summary())
gan.train_gan(X_train)
