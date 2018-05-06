from data_loader import ImageLoader
from models import *

RECONST_DIR = "../reconst"

def main():
    n_epoch = 50

    train_img_loader = ImageLoader("../data/train", "../data/train.npz")
    train_imgs, paths = train_img_loader.getData()

    test_img_loader = ImageLoader("../data/test", "../data/test.npz")
    test_imgs, paths = test_img_loader.getData()
    
    #plt_reconst_img_callback = PlotReconstImages(test_imgs, RECONST_DIR)

    vae = VAE(train_imgs, test_imgs)
    vae.train(n_epoch=n_epoch)


if __name__ == '__main__':
    main()
