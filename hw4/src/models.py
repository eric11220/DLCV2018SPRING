import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.callbacks import Callback
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

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional


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


def autoencoder(input_shape):
    input_img, encoded = encoder(input_shape)
    decoded = decoder(encoded)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.summary()
    return autoencoder


def sampling(args, latent_dim=512, epsilon_std=1.0):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def VAE(input_shape, latent_dim=512):
    input_img, encoded = encoder(input_shape)
    encoded_flattened = Flatten()(encoded)

    # Variational Sampling
    z_mean = Dense(latent_dim)(encoded_flattened)
    z_log_var = Dense(latent_dim)(encoded_flattened)
    z = Lambda(sampling, arguments={'latent_dim': latent_dim}, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_flattened_input = Dense(1024)(z)
    decoder_input = Reshape((4, 4, 64))(decoder_flattened_input)
    print(decoder_input)

    decoded = decoder(decoder_input)
    vae = Model(input_img, decoded)
    vae.compile(optimizer='adam', loss='mean_squared_error')
    vae.summary()
    return vae
