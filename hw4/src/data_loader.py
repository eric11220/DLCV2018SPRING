import numpy as np
import os

from scipy import misc


class ImageLoader():
    def __init__(self, data_dir, presaved_path=None):
        self._data_dir = data_dir

        if data_dir[-1] == '/':
            data_dir = data_dir[:-1]

        self._presaved_path = presaved_path
        self._attr_path = "%s.csv" % data_dir

        self._info = self._load_data()

    def _load_data(self):
        if self._presaved_path is not None and os.path.isfile(self._presaved_path):
            info = np.load(self._presaved_path)
        else:
            paths, imgs = [], []
            for path in os.listdir(self._data_dir):
                paths.append(path)
                path = os.path.join(self._data_dir, path)

                img = misc.imread(path)
                imgs.append(img)

            paths = np.asarray(paths)
            imgs = np.asarray(imgs)

            attr_dict = self._load_attrs(paths)
            info = {**{'paths': paths, 'imgs': imgs}, **attr_dict}
            if self._presaved_path is not None:
                np.savez(self._presaved_path, **info)

        return info

    def _load_attrs(self, paths):
        attr_dict = {}
        with open(self._attr_path, "r") as inf:
            header = inf.readline().strip().split(',')[1:]
            for line in inf:
                info = line.strip().split(',')
                path, attrs = info[0], info[1:]
                attr_dict[path] = attrs

        attrs = np.asarray([attr_dict[p] for p in paths], dtype=np.float32)
        attr_dict = {name: attrs[:, idx] for idx, name in enumerate(header)}
        return attr_dict

    def getData(self, attr="Smiling"):
        return (self._info['imgs'].astype(np.float32) - 127.5) / 127.5, self._info['paths'], self._info[attr]
