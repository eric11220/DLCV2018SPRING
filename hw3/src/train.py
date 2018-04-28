from keras.utils.np_utils import to_categorical

from models import *
from data_loader import DataLoader
from mean_iou_evaluate import *
DATA_DIR = "../data"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def debug_load_data():
    train_dl = DataLoader(os.path.join(DATA_DIR, "train"))
    val_dl = DataLoader(os.path.join(DATA_DIR, "validation"))
    return train_dl, val_dl


def run_testing(model, val_dl, seg_img_dir, batch_num=32):
    epoch_finish, batch_X, _, names = val_dl.next_batch(batch_size=batch_size)
    while epoch_finish is False:
        seg_imgs_into_dir(model, batch_X, names, seg_ing_dir)
        epoch_finish, batch_X, _, names = val_d.next_batch(batch_size=batch_size)


def seg_imgs_into_dir(model, X, names, seg_img_dir):
    if not os.path.isdir(seg_img_dir):
        os.makedirs(seg_img_dir, exist_ok=True)

    for x, name in zip(X, names):
        pred = model.predict(x)

        path = os.path.join(seg_img_dir, "%s_mask.png" % name)
        misc.imsave(path, pred)


def main():
    train_dl = DataLoader(os.path.join(DATA_DIR, "train"))
    val_dl = DataLoader(os.path.join(DATA_DIR, "validation"))

    n_epoch, batch_size = 10, 3

    fcn = None
    for _ in range(n_epoch):
        epoch_finish, batch_X, batch_y, _ = train_dl.next_batch(batch_size=batch_size)
        while epoch_finish is False:
            if fcn is None:
                fcn = FCN_Vgg16_32s(input_shape=batch_X[0].shape)

            loss = fcn.train_on_batch(batch_X, batch_y)
            epoch_finish, batch_X, batch_y, _ = train_dl.next_batch(batch_size=batch_size)


if __name__ == '__main__':
    main()
