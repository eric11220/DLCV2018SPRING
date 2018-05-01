import numpy as np
import os
from scipy import misc


class DataLoader():
    def __init__(self, data_dir, npz_path=None, normalize=False):
        self._X, self._y, self._names = self._load_data(data_dir, npz_path)

        self._normalize = normalize
        if normalize is True:
            self._deduct = np.asarray([0.485, 0.456, 0.406])
            self._dividend = np.asarray([0.229, 0.224, 0.225])

        self._start = 0
        self._num_data = self._X.shape[0]

        self._shuffle()

    def _raw_masks_to_cate(self, masks):

        n, h, w, _ = masks.shape
        new_masks = np.zeros((n, h, w), dtype=np.uint8)

        masks = (masks >= 128).astype(int)
        masks = np.dot(masks, np.asarray([4, 2, 1]))

        new_masks[masks == 3] = 0  # (Cyan: 011) Urban land
        new_masks[masks == 6] = 1  # (Yellow: 110) Agriculture land
        new_masks[masks == 5] = 2  # (Purple: 101) Rangeland
        new_masks[masks == 2] = 3  # (Green: 010) Forest land
        new_masks[masks == 1] = 4  # (Blue: 001) Water
        new_masks[masks == 7] = 5  # (White: 111) Barren land
        new_masks[masks == 0] = 6  # (Black: 000) Unknown

        return new_masks
    
    def _load_data(self, data_dir, npz_path=None):
        # Remove trailing slash
        if data_dir[-1] == '/':
            data_dir = data_dir[:-1]
    
        # Construct a pre-save npz path using data dir name
        if npz_path is None:
            basename = os.path.basename(data_dir)
            upper_dir = os.path.dirname(data_dir)
            npz_path = os.path.join(upper_dir, "%s.npz" % basename)
    
        # Load pre-saved images and masks
        if os.path.isfile(npz_path):
            info = np.load(npz_path)
            X, y, img_names = info['arr_0'], info['arr_1'], info['arr_2']
        else:
            X, y, img_names = [], [], []
            for cnt, img_name in enumerate(os.listdir(data_dir)):
                name, ext = os.path.splitext(img_name)
                if ext != '.jpg':
                    continue

                if cnt % 100 == 0:
                    print("Processed %d sat images and masks..." % cnt);
    
                img_id, _ = name.split("_")
                mask_name = "%s_mask.png" % img_id
    
                sat_img_path = os.path.join(data_dir, img_name)
                mask_path = os.path.join(data_dir, mask_name)
    
                X.append(misc.imread(sat_img_path))
                y.append(misc.imread(mask_path))

                idx, _ = name.split('_')
                img_names.append(idx)
    
            X = np.asarray(X)
            y = np.asarray(y)
            y = self._raw_masks_to_cate(y)
            img_names = np.asarray(img_names)
    
            np.savez(npz_path, X, y, img_names)
    
        print("X shape: %s, y shape: %s" % (X.shape, y.shape))
        return X, y, img_names

    def _cate_mask_to_one_hot(self, masks, n_classes=7):
        n, h, w = masks.shape
        one_hots = []
        for mask in masks:
            one_hot = (np.arange(n_classes) == mask[...,None]).astype(int)
            one_hots.append(one_hot)
        one_hots = np.asarray(one_hots)
        return one_hots

    def _shuffle(self):
        ind = np.random.permutation(self._num_data)
        self._X = self._X[ind]
        self._y = self._y[ind]
        self._names = self._names[ind]
        self._start = 0

    def next_batch(self, batch_size=32):
        if self._start >= self._num_data:
            self._shuffle()
            return True, None, None, None
        else:
            start, end = self._start, self._start + batch_size
            self._start += batch_size
            X = np.array(self._X[start:end])
            y = self._cate_mask_to_one_hot(self._y[start:end])
            names = self._names[start:end]

            if self._normalize is True:
                X = X.astype(np.float32) / 255.
                X = (X - self._deduct) / self._dividend
            return False, X, y, names


    def get_data(self, n_data=100):
        X = self._X[:n_data]
        y = self._y[:n_data]
        names = self._names[:n_data]

        if self._normalize is True:
            X = X.astype(np.float32) / 255.
            X = (X - self._deduct) / self._dividend

        return X, y, names
