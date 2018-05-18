from __future__ import print_function, division
import numpy as np
import os
import sys
import tensorflow as tf

from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

from matplotlib import pyplot as plt
from scipy.misc import imread, imsave


class ACGAN():
    def __init__(self, imgs, labels, n_classes=2, lr=2e-4, beta_1=0.5, ac_loss_weight=1):
        self.imgs = imgs
        self.img_rows, self.img_cols, self.channels = imgs[0].shape
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.labels = labels
        self.n_classes = n_classes

        np.random.seed(820)
        self.noise = np.random.normal(0, 1, (50, 100))

        # Use ADAM for optimizer
        optimizer = Adam(lr, beta_1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                                   loss_weights=[1., ac_loss_weight],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        plot_model(self.discriminator, to_file="acgan_discriminator.png", show_shapes=True)

        # Build the generator
        self.generator = self.build_generator()
        plot_model(self.generator, to_file="acgan_generator.png", show_shapes=True)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100+1,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, aux = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, [valid, aux])
        self.combined.compile(
                loss=['binary_crossentropy', 'categorical_crossentropy'],
                loss_weights=[1., ac_loss_weight],
                optimizer=optimizer)

    def build_generator(self, gf_dim=128, k=3, latent_dim=100):
        model = Sequential()
        model.add(Dense(input_dim=latent_dim + 1, output_dim=1024))
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

        print("Generator")
        model.summary()
        plot_model(model, to_file='generator.png', show_shapes=True)
        return model

    def build_discriminator(self, df_dim=32, k=3, droprate=0.25):
        model = Sequential()

        model.add(Conv2D(df_dim, k, strides=(2, 2), padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))
        model.add(Dropout(droprate))

        model.add(Conv2D(df_dim * 2, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))
        model.add(Dropout(droprate))

        model.add(Conv2D(df_dim * 4, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))
        model.add(Dropout(droprate))

        model.add(Conv2D(df_dim * 8, k, strides=(2, 2), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=.2))
        model.add(Dropout(droprate))
        plot_model(model, to_file="acgan_discriminator_middle.png", show_shapes=True)

        model.add(Flatten())

        input_img = Input(shape=self.img_shape)
        features = model(input_img)
        plot_model(model, to_file='shared_weight_model.png', show_shapes=True)

        fake_or_real = Dense(1, activation='sigmoid')(features)
        aux = Dense(self.n_classes, activation='softmax')(features)

        print("Discriminator")
        model = Model(input_img, [fake_or_real, aux])
        model.summary()
        plot_model(model, to_file='discriminator.png', show_shapes=True)
        return model

    def train(self, steps, batch_size=64, sample_interval=50):
        half_batch = int(batch_size / 2)

        # Do not maximize accuracy on attributes of generated images
        # Set corresponding weights to zero
        d_sample_weight = [np.ones(half_batch), np.zeros(half_batch)]

        attr_accus_real, attr_accus_fake, accus_real, accus_fake = [], [], [], []
        for step in range(steps):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], half_batch)
            imgs = self.imgs[idx]
            labels = self.labels[idx]

            # Generate a half batch of new images with sampled labels from p_c
            noise = np.random.normal(0, 1, (half_batch, 100))
            sampled_half_labels = np.random.randint(0, self.n_classes, half_batch)
            noise_with_labels = np.concatenate((noise, sampled_half_labels.reshape(-1, 1)), axis=-1)

            gen_imgs = self.generator.predict(noise_with_labels)

            # Train the discriminator
            batch_imgs = np.vstack((imgs, gen_imgs))
            batch_labels = np.concatenate((labels, sampled_half_labels))
            batch_labels = to_categorical(batch_labels, num_classes=self.n_classes)

            # Random flip real-fake labels
            do_flip = np.random.uniform(1., 1., batch_size) > 1.
            real_fake_label = np.concatenate((np.ones(half_batch), np.zeros(half_batch))) 
            real_fake_label = np.logical_xor(real_fake_label, do_flip)

            d_loss_real = self.discriminator.train_on_batch(
                    imgs,
                    [np.ones(half_batch), batch_labels[:half_batch]])

            d_loss_fake = self.discriminator.train_on_batch(
                    gen_imgs,
                    [np.zeros(half_batch), batch_labels[half_batch:]])
                    #sample_weight=d_sample_weight)

            d_loss = d_loss_real + d_loss_fake
            real_fake_accu = d_loss[3]
            attr_accu = d_loss[4]

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))
            sampled_labels = np.random.randint(0, self.n_classes, batch_size)
            noise_with_labels = np.concatenate((noise, sampled_labels.reshape(-1, 1)), axis=-1)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(
                    noise_with_labels,
                    [valid_y, to_categorical(sampled_labels, num_classes=self.n_classes)])

            # Plot the progress
            print ("%d [D loss: %f, fake-real acc.: %.2f%%, attr acc.: %.2f%%] [G loss: %f]"
                    % (step, d_loss[0], 100*real_fake_accu, 100*attr_accu, g_loss[0]))

            attr_accus_real.append(d_loss_real[4])
            attr_accus_fake.append(d_loss_fake[4])
            accus_real.append(d_loss_real[3])
            accus_fake.append(d_loss_fake[3])

            # If at save interval => save generated image samples
            if step % sample_interval == 0:
                os.makedirs("../models/acgan", exist_ok=True)
                self.generator.save("../models/acgan/step%d.h5" % step)
                self.sample_images(step)

        return {'Training Loss of Attribute Classification': [attr_accus_real, attr_accus_fake],
                'Accuracy of Discriminator': [accus_real, accus_fake]}

    def sample_images(self, step, r=10, c=10, sample_img_dir="../acgan_images"):
        os.makedirs(sample_img_dir, exist_ok=True)

        half_sample = int(r * c / 2)
        ones = np.ones(half_sample)
        zeros = np.zeros(half_sample)
        labels = np.concatenate((ones, zeros), axis=-1)

        noise_with_labels = np.concatenate((np.concatenate((self.noise, self.noise), axis=0), labels.reshape(-1, 1)), axis=-1)
        gen_imgs = self.generator.predict(noise_with_labels)

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

        fig.savefig(os.path.join(sample_img_dir, "face_%d.png" % step))
        plt.close()
