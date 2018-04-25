import numpy as np
import os
from scipy import misc


class DataLoader():
    def __init__(self, data_dir, npz_path=None):
        self._X, self._y = self._load_data(data_dir, npz_path)
        self._y = self._raw_masks_to_cate(self._y)

        self._start = 0
        self._num_data = self._X.shape[0]

        self._shuffle()

    def _raw_masks_to_cate(self, masks, n_classes=21):
        masks = (masks >= 128).astype(int)
        masks = 4 * masks[:, :, :, 0] + 2 * masks[:, :, :, 1] + masks[:, :, :, 2]
        return masks
    
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
            X, y = info['arr_0'], info['arr_1']
        else:
            X, y = [], []
            for cnt, img_name in enumerate(os.listdir(data_dir)):
                if cnt % 100 == 0:
                    print("Processed %d sat images and masks..." % cnt);
    
                name, ext = os.path.splitext(img_name)
                if ext != '.jpg':
                    continue
    
                img_id, _ = name.split("_")
                mask_name = "%s_mask.png" % img_id
    
                sat_img_path = os.path.join(data_dir, img_name)
                mask_path = os.path.join(data_dir, mask_name)
    
                X.append(misc.imread(sat_img_path))
                y.append(misc.imread(mask_path))
    
            X = np.asarray(X)
            y = np.asarray(y)
    
            np.savez(npz_path, X, y)
    
        print("X shape: %s, y shape: %s" % (X.shape, y.shape))
        return X, y

    def _cate_mask_to_one_hot(self, masks, n_classes=21):
        n, h, w = masks.shape
        one_hots = []
        for mask in masks:
            one_hot = (np.arange(n_classes) == mask[...,None]-1).astype(int)
            one_hots.append(one_hot)
        one_hots = np.asarray(one_hots)
        return one_hots

    def _shuffle(self):
        ind = np.random.permutation(self._num_data)
        self._X = self._X[ind]
        self._y = self._y[ind]

    def next_batch(self, batch_size=32):
        if self._start > self._num_data:
            self._shuffle()
            return True, None, None
        else:
            start, end = self._start, self._start + batch_size
            self._start += batch_size
            X = self._X[start:end]
            y = self._cate_mask_to_one_hot(self._y[start:end])
            return False, X, y
