from keras.layers import Reshape, LeakyReLU, Dense
from keras.models import Sequential


# Generator model
class Generator(object):
    def __init__(self):
        self.model = self.create_generator()

    def create_generator(self) -> Sequential:
        model = Sequential()

        model.add(Dense(256, input_dim=100))
        model.add(LeakyReLU(0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(0.2))
        model.add(Dense(784, activation="tanh"))
        model.add(Reshape((28, 28, 1)))

        return model

    def summary(self):
        return self.model.summary()
