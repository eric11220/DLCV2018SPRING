import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys

from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from data_loader import ImageLoader
from vae import VAE


def plot_losses(losses, path):
    plt.clf()

    n_subplot = len(losses)
    plt.figure(figsize=(12, 8))
    for idx, (name, loss) in enumerate(losses.items()):
        plt.subplot(1, n_subplot, idx+1)
        plt.plot(range(len(loss)), loss)
        plt.xlabel("Training Epoch")
        plt.title(name)

    plt.tight_layout()
    plt.savefig(path)

def plot_vae_random_images(decoder, path):
    np.random.seed(820)
    keep_ind = [18, 23, 30, 65, 66, 77, 79, 96,
                100, 120, 122, 207, 215, 217, 331, 350,
                389, 390, 407, 421, 445, 447, 523, 534,
                600, 629, 665, 685, 687, 698, 705, 731]
    random_z = np.random.normal(0, 4, (800, 512))
    random_z = random_z[keep_ind, :]

    imgs = decoder.predict(random_z)
    imgs = imgs * 127.5 + 127.5
    imgs = imgs.astype(np.uint8)

    plt.clf()
    plot_img = np.zeros((256, 512, 3), dtype=np.uint8)
    for fid, img in enumerate(imgs):
        y_idx = int(fid / 8)
        x_idx = fid % 8
        plot_img[y_idx*64:(y_idx+1)*64, x_idx*64:(x_idx+1)*64, :] = img

    plt.imshow(plot_img)
    plt.axis('off')
    path = os.path.join(path)

    plt.tight_layout()
    plt.savefig(path)

def plot_vae_reconst_images(encoder, decoder, imgs, path):
    mean = encoder.predict(imgs)
    reconst = decoder.predict(mean)
    reconst = reconst * 127.5 + 127.5
    reconst = reconst.astype(np.uint8)

    imgs = imgs * 127.5 + 127.5
    imgs = imgs.astype(np.uint8)

    plt.clf()
    plot_img = np.zeros((128, 640, 3), dtype=np.uint8)
    for fid, (reconst_img, img) in enumerate(zip(reconst, imgs)):
        plot_img[:64, fid*64:(fid+1)*64, :] = img
        plot_img[64:, fid*64:(fid+1)*64, :] = reconst_img

    plt.axis('off')
    plt.imshow(plot_img)

    plt.tight_layout()
    plt.savefig(path)

def plot_tsne(encoder, imgs, attrs, path):
    feats = encoder.predict(imgs)
    feats_2d = TSNE(n_components=2, random_state=820).fit_transform(feats)

    pos = feats_2d[attrs==1]
    neg = feats_2d[attrs==0]
    plt.clf()
    plt.scatter(pos[:, 0], pos[:, 1], color='r', label="Smiling")
    plt.scatter(neg[:, 0], neg[:, 1], color='b', label="Not Smiling")
    plt.title("Result")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)

def plot_dcgan_images(generator, path):
    np.random.seed(820)
    z = np.random.normal(0, 1, (32, 100))

    imgs = generator.predict(z)
    imgs = imgs * 127.5 + 127.5
    imgs = imgs.astype(np.uint8)

    plt.clf()
    plot_img = np.zeros((256, 512, 3), dtype=np.uint8)
    for fid, img in enumerate(imgs):
        y_idx = int(fid / 8)
        x_idx = fid % 8
        plot_img[y_idx*64:(y_idx+1)*64, x_idx*64:(x_idx+1)*64, :] = img

    plt.imshow(plot_img)
    plt.axis('off')
    path = os.path.join(path)

    plt.tight_layout()
    plt.savefig(path)

def plot_acgan_images(generator, path):
    np.random.seed(820)
    noise = np.random.normal(0, 1, (50, 100))
    keep_ind = [3, 4, 5, 8, 10, 15, 27, 36, 38, 39]

    labels = np.zeros((10, 1))
    inputs = np.concatenate((noise[keep_ind, :], labels), axis=-1)
    not_smile = generator.predict(inputs)
    not_smile = not_smile * 127.5 + 127.5

    labels = np.ones((10, 1))
    inputs = np.concatenate((noise[keep_ind, :], labels), axis=-1)
    smile = generator.predict(inputs)
    smile = smile * 127.5 + 127.5

    plt.clf()
    plot_imgs = np.zeros((128, 640, 3), dtype=np.uint8)
    for idx, (img, img_smile) in enumerate(zip(not_smile, smile)):
        plot_imgs[:64, idx*64:(idx+1)*64, :] = img
        plot_imgs[64:, idx*64:(idx+1)*64, :] = img_smile

    plt.imshow(plot_imgs)
    plt.axis('off')
    plt.title("Not Smiling")
    plt.xlabel("Smiling")

    plt.tight_layout()
    plt.savefig(path)

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    ## VAE Part
    # Learning curves
    losses = np.load("stats/vae.npy")
    losses = losses.item()
    plot_losses(losses, os.path.join(output_dir, 'fig1_2.jpg'))

    # Testing images reconstruction
    test_img_loader = ImageLoader(os.path.join(input_dir, "test"))
    test_imgs, _, test_attrs = test_img_loader.getData()

    # Loaad VAE decoder
    vae_decoder = load_model("models/vae_decoder.h5", custom_objects={'KLDivergenceLayer': VAE.KLDivergenceLayer})
    vae_encoder = load_model("models/vae_encoder.h5", custom_objects={'KLDivergenceLayer': VAE.KLDivergenceLayer})

    # Plot reconstructed images
    path = os.path.join(output_dir, "fig1_3.png")
    plot_vae_reconst_images(vae_encoder, vae_decoder, np.array(test_imgs[10:20]), path)

    # Plot random images
    path = os.path.join(output_dir, "fig1_4.png")
    plot_vae_random_images(vae_decoder, path)

    # Plot TSNE image
    path = os.path.join(output_dir, "fig1_5.png")
    plot_tsne(vae_encoder, test_imgs, test_attrs, path)

    ## DCGAN Part
    # Learning curves
    losses = np.load("stats/dcgan.npy")
    losses = losses.item()
    plot_losses(losses, os.path.join(output_dir, 'fig2_2.jpg'))

    # Random images
    dcgan = load_model("models/dcgan.h5")
    plot_dcgan_images(dcgan, os.path.join(output_dir, "fig2_3.png"))

    ## ACGAN Part
    # Learning curves
    losses = np.load("stats/acgan.npy")
    losses = losses.item()

    real_accu , fake_accu = losses['Accuracy of Discriminator']
    real_accu = real_accu[::1000]
    fake_accu = fake_accu[::1000]
    attr_real_accu , attr_fake_accu = losses['Training Loss of Attribute Classification']
    attr_real_accu = attr_real_accu[::1000]
    attr_fake_accu = attr_fake_accu[::1000]

    plt.clf()
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Accuracy of Discriminator")
    real, = plt.plot(range(len(real_accu)), real_accu, label='real')
    fake, = plt.plot(range(len(fake_accu)), fake_accu, label='fake')
    plt.legend(handles=[real, fake])

    plt.subplot(1, 2, 2)
    plt.title("Training Loss of Attribute Classification")
    real, = plt.plot(range(len(real_accu)), attr_real_accu, label='real')
    fake, = plt.plot(range(len(fake_accu)), attr_fake_accu, label='fake')
    plt.legend(handles=[real, fake])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_2.png"))

    # Random smiling pairs
    acgan = load_model("models/acgan.h5")
    plot_acgan_images(acgan, os.path.join(output_dir, "fig3_3.png"))


if __name__ == '__main__':
    main()
