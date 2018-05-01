import shutil
import tempfile
from keras.utils.np_utils import to_categorical

from models import *
from data_loader import DataLoader
from mean_iou_evaluate import *
from scipy import misc

from mean_iou_evaluate import *
DATA_DIR = "../data"
SEG_TRAIN_MASK_DIR = "../results/train"
SEG_TEST_MASK_DIR = "../results/test"


def debug_load_data(normalize=False):
    train_dl = DataLoader(os.path.join(DATA_DIR, "train"), normalize=normalize)
    val_dl = DataLoader(os.path.join(DATA_DIR, "validation"), normalize=normalize)
    return train_dl, val_dl


def cate_to_colors(masks, n_classes=7):
    color_dict = {  0: np.asarray([0, 255, 255], dtype=np.uint8),
                    1: np.asarray([255, 255, 0], dtype=np.uint8),
                    2: np.asarray([255, 0, 255], dtype=np.uint8),
                    3: np.asarray([0, 255, 0], dtype=np.uint8),
                    4: np.asarray([0, 0, 255], dtype=np.uint8),
                    5: np.asarray([255, 255, 255], dtype=np.uint8),
                    6: np.asarray([0, 0, 0], dtype=np.uint8)}

    n, h, w = masks.shape
    final_masks = [color_dict[cate] for cate in masks.reshape(-1)]
    final_masks = np.asarray(final_masks).reshape((n, h, w, 3))
    '''
    final_masks = np.zeros((n, h, w, 3), dtype=np.uint8)
    for idx in range(n):
        for row in range(h):
            for col in range(w):
                final_masks[idx, row, col, :] = color_dict[masks[idx, row, col]]
    '''
    return final_masks


def run_testing(model, val_dl, seg_img_dir, batch_size=3, truth_dir="../data/validation"):
    epoch_finish, batch_X, _, names = val_dl.next_batch(batch_size=batch_size)
    while epoch_finish is False:
        seg_imgs_into_dir(model, batch_X, names, seg_img_dir)
        epoch_finish, batch_X, _, names = val_dl.next_batch(batch_size=batch_size)

    pred = read_masks(seg_img_dir)
    labels = read_masks(truth_dir)
    return mean_iou_score(pred, labels)


def get_train_iou(model, train_dl, seg_img_dir, batch_size=3, num_plt=100, truth_dir="../data/train"):
    if os.path.isdir(seg_img_dir):
        shutil.rmtree(seg_img_dir)
    X, _, names = train_dl.get_data(num_plt)

    start = 0
    while start <= num_plt:
        end = start + batch_size
        seg_imgs_into_dir(model, X[start:end], names[start:end], seg_img_dir)
        start += batch_size

    tmp_dir = tempfile.mkdtemp()
    for name in names:
        from_path = os.path.join(truth_dir, "%s_mask.png" % name)
        to_path = os.path.join(tmp_dir, "%s_mask.png" % name)
        shutil.copyfile(from_path, to_path)

    pred = read_masks(seg_img_dir)
    labels = read_masks(tmp_dir)
    shutil.rmtree(tmp_dir)
    return mean_iou_score(pred, labels)


def seg_imgs_into_dir(model, X, names, seg_img_dir):
    if not os.path.isdir(seg_img_dir):
        os.makedirs(seg_img_dir, exist_ok=True)

    preds = model.predict(X)
    preds = np.argmax(preds, axis=-1)
    masks = cate_to_colors(preds)

    for img, mask, name in zip(X, masks, names):
        path = os.path.join(seg_img_dir, "%s_mask.png" % name)
        misc.imsave(path, mask)


def main(train_dl, val_dl):
#def main():
    n_epoch, batch_size, max_val_iou = 10, 3, 0.
    normalize = True
    saved_model_dir = "../models"
    os.makedirs(saved_model_dir, exist_ok=True)

    '''
    train_dl = DataLoader(os.path.join(DATA_DIR, "train"), normalize=normalize)
    val_dl = DataLoader(os.path.join(DATA_DIR, "validation"), normalize=normalize)
    '''

    fcn = FCN_Vgg16_32s(input_shape=(512, 512, 3))
    for epoch_idx in range(n_epoch):
        batch_cnt = 0
        epoch_finish, batch_X, batch_y, names = train_dl.next_batch(batch_size=batch_size)

        while epoch_finish is False:
            if batch_cnt % 100 == 0:
                print("Processed %d batches..." % batch_cnt)

            loss = fcn.train_on_batch(batch_X, batch_y)
            epoch_finish, batch_X, batch_y, _ = train_dl.next_batch(batch_size=batch_size)
            batch_cnt += 1

        get_train_iou(fcn, train_dl, SEG_TRAIN_MASK_DIR)
        val_mean_iou = run_testing(fcn, val_dl, SEG_TEST_MASK_DIR, batch_size=batch_size)
        if val_mean_iou > max_val_iou:
            model_path = os.path.join(saved_model_dir, "%.4f.h5" % val_mean_iou)
            fcn.save(model_path)
            print("IOU %.4f better than maxx IOU: %.4f, saving model to %s..." % (val_mean_iou, max_val_iou, model_path))

            max_val_iou = val_mean_iou


if __name__ == '__main__':
    main()
