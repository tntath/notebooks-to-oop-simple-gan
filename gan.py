import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from discriminator import Discriminator
from generator import Generator

# GAN model


class Gan:
    def __init__(
        self,
    ):
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.model = self.create_gan()

    def create_gan(self) -> Model:
        self.discriminator.model.trainable = False
        gan_input = Input(shape=(100,))
        x = self.generator.model(gan_input)
        gan_output = self.discriminator.model(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))
        return gan

    def summary(self):
        return self.model.summary()

    def train_gan(self, X_train, epochs=40000, batch_size=128):
        for e in range(epochs):
            generator = self.generator.model
            discriminator = self.discriminator.model
            # Train the discriminator
            real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            generated_images = generator.predict(noise)
            X = np.concatenate([real_images, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # label smoothing
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            # Train the generator
            noise = np.random.normal(0, 1, size=[batch_size, 100])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = self.model.train_on_batch(noise, y_gen)

            # Print losses
            if e % 1000 == 0:
                print("Discriminator loss: ", d_loss, ", Generator loss: ", g_loss)

        self.save_images()

    def save_images(self):
        noise = np.random.normal(0, 1, size=(100, 100))
        generated_images = self.generator.model.predict(noise)
        matplotlib.use("TkAgg")
        plt.figure(figsize=(10, 10))

        for i, image in enumerate(generated_images):
            plt.subplot(10, 10, i + 1)
            if generated_images.shape[3] == 1:
                plt.imshow(image.reshape((28, 28)), cmap="gray")
            else:
                plt.imshow(image.reshape((28, 28, 3)))
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("gan.png")
        plt.close("all")
