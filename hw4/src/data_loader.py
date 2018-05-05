import numpy as np
import os

from scipy import misc


class ImageLoader():
    def __init__(self, data_dir, presaved_path):
        self._data_dir = data_dir
        self._presaved_path = presaved_path
        self._imgs, self._paths = self._load_data()

    def _load_data(self):
        if os.path.isfile(self._presaved_path):
            info = np.load(self._presaved_path)
            imgs, paths = info['imgs'], info['paths']
        else:
            paths, imgs = [], []
            for path in os.listdir(self._data_dir):
                paths.append(path)
                path = os.path.join(self._data_dir, path)

                img = misc.imread(path)
                imgs.append(img)

            paths = np.asarray(paths)
            imgs = np.asarray(imgs)
            np.savez(self._presaved_path, **{'imgs': imgs, 'paths': paths})

        return imgs, paths

    def getData(self):
        return self._imgs.astype(np.float32) / 255., self._paths
