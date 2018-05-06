from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import tensorflow as tf
from scipy.misc import imread, imsave

import sys
import os

import numpy as np


class GAN():
    def __init__(self, imgs, lr=2e-4, beta_1=0.5):
        self.imgs = imgs
        self.img_rows, self.img_cols, self.channels = imgs[0].shape
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(lr, beta_1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024, activation='relu'))
        model.add(Reshape((4, 4, 64), input_shape=(64*4*4,)))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(3, (3, 3), padding='same', activation='tanh'))
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=self.img_shape))
        model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    def train(self, epochs, batch_size=64, sample_interval=50):

        half_batch = int(batch_size / 2)
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], half_batch)
            imgs = self.imgs[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch, r=6, c=8, sample_img_dir="../images"):
        os.makedirs(sample_img_dir, exist_ok=True)

        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        # NOTE: This depends on how DataLoader processes images
        gen_imgs *= 255
        gen_imgs = gen_imgs.astype(np.uint8)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig(os.path.join(sample_img_dir, "face_%d.png" % epoch))
        plt.close()
