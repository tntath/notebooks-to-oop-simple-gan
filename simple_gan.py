from keras.datasets import mnist
import numpy as np

#Load the dataset
(X_train, _), (_,_) = mnist.load_data()

#Normalize the images -convert pixel values from [0,255] to [-1, 1]
X_train = X_train / 127.5 -1

X_train = np.expand_dims(X_train, axis = 3)

print(X_train.shape)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import LeakyReLU
from keras.optimizers.legacy import Adam

#Discriminator model
def create_discriminator():
    model = Sequential()

    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

discriminator = create_discriminator()
discriminator.summary()

from keras.layers import Reshape
from keras.layers import Dense

#Generator model
def create_generator():
    model = Sequential()

    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(784, activation="tanh"))
    model.add(Reshape((28,28,1)))

    return model

generator = create_generator()
generator.summary()

from keras.models import Model
from keras.layers import Input

#GAN model

def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

gan = create_gan(discriminator, generator)
gan.summary()

def train_gan(gan, generator, discriminator, epochs=40000, batch_size=128):
    for e in range(epochs):
        #Train the discriminator
        real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        noise = np.random.normal(0,1, size=[batch_size, 100])
        generated_images = generator.predict(noise)
        X = np.concatenate([real_images, generated_images])
        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size] = 0.9 #label smoothing
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)

        #Train the generator
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gen)

        # Print losses
        if e%1000 == 0:
            print("Discriminator loss: ", d_loss, ", Generator loss: ", g_loss)

train_gan(gan, generator, discriminator)