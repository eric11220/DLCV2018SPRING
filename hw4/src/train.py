import argparse
from data_loader import ImageLoader
from models import *

RECONST_DIR = "../reconst"
def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model to train", choices=["vae", "dcgan", "acgan"])
    return parser.parse_args()


def train_vae(train_imgs, test_imgs, n_epoch=50):
    #plt_reconst_img_callback = PlotReconstImages(test_imgs, RECONST_DIR)
    vae = VAE(train_imgs, test_imgs)
    vae.train(n_epoch=n_epoch)


def train_dcgan(imgs, n_epoch=100000):
    dcgan = DCGAN(imgs)
    dcgan.train(n_epoch, sample_interval=500)


def train_acgan(imgs, attrs):
    pass


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
