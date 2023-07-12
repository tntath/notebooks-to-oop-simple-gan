from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import LeakyReLU
from keras.optimizers.legacy import Adam


# Discriminator model
class Discriminator(object):
    def __init__(self, width=28, height=28, channels=1):
        self.width = width
        self.height = height
        self.channels = channels
        self.model = self.create_discriminator()

    def create_discriminator(self) -> Sequential:
        model = Sequential()

        model.add(Flatten(input_shape=(self.width, self.height, self.channels)))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(0.0002, 0.5),
            metrics=["accuracy"],
        )
        return model

    def summary(self):
        return self.model.summary()
