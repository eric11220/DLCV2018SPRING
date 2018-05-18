import argparse
import matplotlib
matplotlib.use('Agg')

from data_loader import ImageLoader
from models import *
from matplotlib import pyplot as plt

RECONST_DIR = "../reconst"
STAT_DIR = "../stats"

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model to train", choices=["vae", "dcgan", "acgan"])
    return parser.parse_args()


def train_vae(train_imgs, test_imgs, n_epoch=100):
    vae = VAE(train_imgs, test_imgs)
    losses = vae.train(n_epoch=n_epoch)
    np.save(os.path.join(STAT_DIR, "vae_losses.npy"), losses)


def train_dcgan(imgs, n_step=60000):
    dcgan = DCGAN(imgs)
    losses = dcgan.train(n_step, sample_interval=500)
    np.save(os.path.join(STAT_DIR, "dcgan.npy"), losses)


def train_acgan(imgs, attrs, n_step=60000):
    acgan = ACGAN(imgs, attrs)
    losses = acgan.train(n_step, sample_interval=500)
    np.save(os.path.join(STAT_DIR, "acgan.npy"), losses)


def main():
    args = parse_input()

    train_img_loader = ImageLoader("../data/train", "../data/train.npz")
    train_imgs, _, train_attrs = train_img_loader.getData()

    test_img_loader = ImageLoader("../data/test", "../data/test.npz")
    test_imgs, _, test_attrs = test_img_loader.getData()

    if args.model == "vae":
        train_vae(train_imgs, test_imgs)
    elif args.model == "dcgan":
        train_dcgan(np.vstack((train_imgs, test_imgs)))
    elif args.model == "acgan":
        train_acgan(np.vstack((train_imgs, test_imgs)), np.concatenate((train_attrs, test_attrs)))


if __name__ == '__main__':
    main()
