from keras.utils.np_utils import to_categorical

from models import *
from data_loader import DataLoader
from mean_iou_evaluate import *
DATA_DIR = "../data"


def main():
    train_dl = DataLoader(os.path.join(DATA_DIR, "train"))
    val_dl = DataLoader(os.path.join(DATA_DIR, "validation"))

    n_epoch, batch_size = 10, 32

    fcn = None
    for _ in range(n_epoch):
        epoch_finish, batch_X, batch_y = train_dl.next_batch()
        print(batch_X.shape)
        print(batch_y.shape)
        input("check shapes")
        while epoch_finish is False:
            if fcn is None:
                fcn = FCN_Vgg16_32s(input_shape=batch_X[0].shape)
            loss = fcn.train_on_batch(batch_X, batch_y)
            print(loss)
            input("check loss")
            epoch_finish, batch_X, batch_y = train_dl.next_batch(batch_size=batch_size)


if __name__ == '__main__':
    main()
