import numpy as np
import os
import sys
from keras.models import load_model
from scipy import misc

batch_size = 1

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
    return final_masks

input_dir = sys.argv[1]
output_dir = sys.argv[2]
model_path = sys.argv[3]

model = load_model(model_path)

imgs, indices = [], []
for path in os.listdir(input_dir):
    if "sat" not in path:
        continue

    img_idx = path.split('_')[0]

    path = os.path.join(input_dir, path)
    img = misc.imread(path)
    imgs.append(img)
    indices.append(img_idx)

imgs = np.asarray(imgs)
indices = np.asarray(indices)

start = 0
while start < len(imgs):
    end = start + batch_size

    masks = model.predict(imgs[start:end])
    masks = np.argmax(masks, axis=-1)
    masks = cate_to_colors(masks)

    for idx, mask in zip(indices[start:end], masks):
        out_path = os.path.join(output_dir, "%s_mask.png" % idx)
        misc.imsave(out_path, mask)

    start += batch_size
