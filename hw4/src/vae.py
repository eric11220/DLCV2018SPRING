import keras.backend as K
from keras import metrics
from keras.callbacks import Callback
from keras.layers import *
from keras.losses import *
from keras.models import *
#from matplotlib import pyplot as plt


'''
class PlotReconstImages(Callback):
    def __init__(self, images, reconst_dir):
        self._images = images
        self._reconst_dir = reconst_dir

    def on_epoch_end(self, epoch, logs={}):
        outdir = os.path.join(self._reconst_dir, "epoch%d" % epoch)
        os.makedirs(outdir, exist_ok=True)

        reconst = self.model.predict(self._images)
        for fid, (reconst_img, img) in enumerate(zip(reconst, self._images)):
            plt.subplot(1, 2, 1)
            plt.imshow(reconst_img)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img)
            plt.axis('off')

            path = os.path.join(out_dir, "%d.png")
            plt.savefig(path)
'''


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
    def __init__(self, train_imgs, test_imgs, latent_dim=512, lamb_kl=1e-5):
        assert train_imgs[0].shape == test_imgs[0].shape
        self.train_imgs = train_imgs
        self.test_imgs= test_imgs
        input_shape = train_imgs[0].shape

        # Convolutional encoder
        input_img, encoded = encoder(input_shape)
        encoded_flattened = Flatten()(encoded)

        # Variational Sampling
        z_mean = Dense(latent_dim)(encoded_flattened)
        z_log_var = Dense(latent_dim)(encoded_flattened)
        z_mean, z_log_var = VAE.KLDivergenceLayer(lamb_kl)([z_mean, z_log_var])
        z = Lambda(self.sampling, arguments={'latent_dim': latent_dim}, output_shape=(latent_dim,))([z_mean, z_log_var])

        # Convolutional decoder
        decoder_flattened_input = Dense(1024)(z)
        decoder_input = Reshape((4, 4, 64))(decoder_flattened_input)
        decoded = decoder(decoder_input)

        # Compile model
        vae = Model(input_img, decoded)
        vae.compile(optimizer='adam', loss='mean_squared_error')
        vae.summary()

        self.model = vae

    def sampling(self, args, latent_dim=512, epsilon_std=1.0):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def train(self, n_epoch=50):
        self.model.fit(self.train_imgs,
                       self.train_imgs,
                       epochs=n_epoch,
                       validation_data=(self.test_imgs, self.test_imgs))


    class KLDivergenceLayer(Layer):
        def __init__(self, lamb_kl, **kwargs):
            self.is_placeholder = True
            self.lamb_kl = lamb_kl
            super(VAE.KLDivergenceLayer, self).__init__(**kwargs)

        def call(self, inputs):
            mu, log_var = inputs
            kl_batch = - .5 * K.sum(1 + log_var -
                                    K.square(mu) -
                                    K.exp(log_var), axis=-1)

            self.add_loss(self.lamb_kl * K.mean(kl_batch), inputs=inputs)
            return inputs
