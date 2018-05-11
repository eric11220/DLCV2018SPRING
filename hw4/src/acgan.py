from __future__ import print_function, division
import numpy as np
import os
import sys
import tensorflow as tf

from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

from matplotlib import pyplot as plt
from scipy.misc import imread, imsave


class ACGAN():
    def __init__(self, imgs, labels, n_classes=2, lr=2e-4, beta_1=0.5):
        self.imgs = imgs
        self.img_rows, self.img_cols, self.channels = imgs[0].shape
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(lr, beta_1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
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

    def build_generator(self, gf_dim=64, k=5, latent_dim):
        model = Sequential()
        model.add(Dense(input_dim=latent_dim, output_dim=1024))
        model.add(Reshape((4, 4, 64), input_shape=(64 * 4 * 4,)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(gf_dim * 4, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(gf_dim * 2, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(gf_dim * 1, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(3, k, strides=(2, 2), padding='same'))
        model.add(Activation('tanh'))

        # Handle input
        latent = Input(shape=(latent_dim, ))
        image_class = Input(shape=(1,), dtype='int32')

        # hadamard product between z-space and a class conditional embedding
        cls = Flatten()(Embedding(num_classes, latent_size, embeddings_initializer='glorot_normal')(image_class))
        h = layers.multiply([latent, cls])

        fake_image = cnn(h)
        model = Model([latent, image_class], fake_image)

        print("Generator")
        model.summary()
        return model

    def build_discriminator(self, df_dim=16, k=5):
        model = Sequential()
        model.add(Conv2D(df_dim, k, strides=(2, 2), padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))

        model.add(Conv2D(df_dim * 2, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))

        model.add(Conv2D(df_dim * 4, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))

        model.add(Conv2D(df_dim * 8, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))
        model.add(Flatten())

        input_img = Input(shape=self.img_shape)
        features = model(input_img)

        fake_or_real = Dense(1, activation='sigmoid')(features)
        aux = Dense(n_class, activation='softmax')(features)

        print("Discriminator")
        model = Model(input_img, [fake_or_real, aux])
        model.summary()
        return model

    def train(self, epochs, batch_size=64, sample_interval=50):
        half_batch = int(batch_size / 2)

        # Do not maximize accuracy on attributes of generated images
        # Set corresponding weights to zero
        gen_sample_weight = [np.ones(half_batch), np.concatenate(np.zeros(half_batch))]

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], half_batch)
            imgs = self.imgs[idx]
            labels = self.labels[idx]

            # Generate a half batch of new images with sampled labels from p_c
            noise = np.random.normal(0, 1, (half_batch, 100))
            sampled_labels = np.random.randint(0, n_classes, half_batch)
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                    imgs,
                    [np.ones((half_batch, 1), labels]))

            d_loss_fake = self.discriminator.train_on_batch(
                    gen_imgs,
                    [np.zeros((half_batch, 1)), sampled_labels],
                    sample_weight=gen_sample_weight)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))],
                    [valid_y, sampled_labels])

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
        gen_imgs = gen_imgs * 127.5 + 127.5
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
