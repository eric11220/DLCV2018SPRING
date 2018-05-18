import keras.backend as K
import matplotlib
matplotlib.use('Agg')

from keras import metrics
from keras.callbacks import Callback
from keras.layers import *
from keras.losses import *
from keras.models import *
from keras.utils import plot_model
from matplotlib import pyplot as plt

class KLoss(Callback):
    def __init__(self, model, images, records):
        self.images = images
        self.model = model
        self.records = records

        mu_out, log_var_out = model.get_layer("z_mean").output, model.get_layer("z_log_var").output
        self.kl_model = K.function([model.input], [mu_out, log_var_out])

    def kl_loss(self):
        mu, log_var = self.kl_model([self.images])
        kl_batch = - .5 * np.sum(1 + log_var -
                                np.square(mu) -
                                np.exp(log_var), axis=-1)

        kl_loss = np.mean(kl_batch)
        self.records.append(kl_loss)

    def on_epoch_end(self, epoch, logs={}):
        self.kl_loss()


class PlotImages(Callback):
    def __init__(self, images, encoder, decoder, random_z, reconst_dir):
        self.images = images
        self.encoder = encoder
        self.decoder = decoder
        self.random_z = random_z
        self.reconst_dir = reconst_dir

    def plot_reconst_images(self, n_img=10):
        imgs = np.array(self.images[10:10+n_img])

        mean = self.encoder.predict(imgs)
        reconst = self.decoder.predict(mean)
        reconst = reconst * 127.5 + 127.5
        reconst = reconst.astype(np.uint8)

        imgs = imgs * 127.5 + 127.5
        imgs = imgs.astype(np.uint8)
        for fid, (reconst_img, img) in enumerate(zip(reconst, imgs)):
            plt.subplot(2, n_img, fid+1)
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(2, n_img, n_img+fid+1)
            plt.imshow(reconst_img)
            plt.axis('off')

        path = os.path.join(self.reconst_dir, "epoch%d_reconst.png" % self.epoch)
        plt.savefig(path)

    def plot_random_images(self):
        imgs = self.decoder.predict(self.random_z)
        imgs = imgs * 127.5 + 127.5
        imgs = imgs.astype(np.uint8)

        for fid, img in enumerate(imgs):
            plt.subplot(10, 10, fid+1)
            plt.imshow(img)
            plt.axis('off')

        path = os.path.join(self.reconst_dir, "epoch%d_random.png" % self.epoch)
        plt.savefig(path)

    def on_epoch_end(self, epoch, logs={}):
        os.makedirs(self.reconst_dir, exist_ok=True)
        self.epoch = epoch
        self.plot_reconst_images()
        self.plot_random_images()


def encoder(input_shape):
    input_img = Input(shape=input_shape)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return input_img, encoded

    # at this point the representation is (4, 4, 64) i.e. 1024-dimensional


def decoder(encoded):
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    return decoded


class AutoEncoder():
    def __init__(self, imgs):
        self.imgs = imgs
        input_shape = imgs[0].shape

        input_img, encoded = encoder(input_shape)
        decoded = decoder(encoded)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.summary()
        self.model = autoencoder


class VAE():
    def __init__(self, train_imgs, test_imgs, latent_dim=512, lamb_kl=1e-4):
        assert train_imgs[0].shape == test_imgs[0].shape
        self.latent_dim = latent_dim
        self.train_imgs = train_imgs
        self.test_imgs= test_imgs
        input_shape = train_imgs[0].shape

        # Convolutional encoder
        input_img, encoded = encoder(input_shape)
        encoded_flattened = Flatten()(encoded)

        # Variational Sampling
        z_mean = Dense(latent_dim, name='z_mean')(encoded_flattened)
        z_log_var = Dense(latent_dim, name='z_log_var')(encoded_flattened)
        z_mean, z_log_var = VAE.KLDivergenceLayer(lamb_kl)([z_mean, z_log_var])
        z = Lambda(self.sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # Build encoder
        latent_encoder = Model(input_img, z_mean, name='encoder')
        self.encoder = latent_encoder
        latent_encoder.summary()
        plot_model(latent_encoder, to_file='vae_encoder.png', show_shapes=True)

        # build decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        decoder_flattened_input = Dense(1024)(latent_inputs)
        decoder_input = Reshape((4, 4, 64))(decoder_flattened_input)
        outputs = decoder(decoder_input)

        latent_decoder = Model(latent_inputs, outputs, name='decoder')
        latent_decoder.summary()
        self.decoder = latent_decoder
        plot_model(latent_decoder, to_file='vae_decoder.png', show_shapes=True)

        # Compile model
        print(latent_encoder(input_img))
        vae_outputs = latent_decoder(z)
        vae = Model(input_img, vae_outputs)
        vae.compile(optimizer='adam', loss='mean_squared_error')
        vae.summary()
        self.model = vae
        plot_model(vae, to_file='vae.png', show_shapes=True)

    def sampling(self, args, epsilon_std=1.0):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def train(self, n_epoch=50):
        np.random.seed(820)
        keep_ind = [3, 13, 14, 16, 26, 29, 38, 43, 57, 63, 73, 74,
                    111, 113, 123, 141, 143, 160, 161, 170, 188, 190, 191, 196,
                    200, 211, 238, 248, 256, 276, 279, 281]

        random_z = np.random.normal(0, 4, (300, self.latent_dim))
        random_z = random_z[keep_ind, :]

        kl_records = []
        hist = self.model.fit(self.train_imgs,
                              self.train_imgs,
                              epochs=n_epoch,
                              callbacks=[PlotImages(self.test_imgs, self.encoder, self.decoder, random_z, "../vae"),
                                         KLoss(self.model, self.test_imgs, kl_records)],
                              validation_data=(self.test_imgs, self.test_imgs))

        self.model.save("../models/vae.h5")
        self.decoder.save("../models/vae_decoder.h5")
        self.encoder.save("../models/vae_encoder.h5")
        return {'MSE': hist.history['val_loss'], 'KLD': kl_records}


    class KLDivergenceLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            self.lamb_kl = 1e-4
            super(VAE.KLDivergenceLayer, self).__init__(**kwargs)

        def call(self, inputs):
            mu, log_var = inputs
            kl_batch = - .5 * K.sum(1 + log_var -
                                    K.square(mu) -
                                    K.exp(log_var), axis=-1)

            self.add_loss(self.lamb_kl * K.mean(kl_batch), inputs=inputs)
            return inputs

        def get_config(self):
            config = {'lamb_kl': self.lamb_kl}
            base_config = super(VAE.KLDivergenceLayer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
