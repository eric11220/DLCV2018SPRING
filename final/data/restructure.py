import argparse
import os
import shutil
import random
from os.path import dirname, basename, join, splitext, abspath, islink
train_dir = "train"

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Generate which csv", default="base")
    parser.add_argument("--kshot", help="K-shot scenario", default=10)
    parser.add_argument("--n-valid", help="Number of validation data", default=60, type=int)

    parser.add_argument("--output-dir", help="Directory for generated csv and image folder", default=".")
    parser.add_argument("--oversampling", help="Oversampling ratio", default=1, type=int)
    return parser.parse_args()

def main():
    args = parse_input()
    random.seed(820)

    img_dir = join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Foundation Class
    cls_idx = 0
    train_paths, val_paths = {}, {}
    for cls in sorted(os.listdir(join(train_dir, "base"))):
        cls_dir = join(train_dir, "base", cls)

        train_subdir = join(cls_dir, "train")
        test_subdir = join(cls_dir, "test")
        img_paths = [join(train_subdir, path) for path in os.listdir(train_subdir)]
        img_paths += [join(test_subdir, path) for path in os.listdir(test_subdir)]
        random.shuffle(img_paths)

        for img_idx, img_path in enumerate(img_paths):
            basepath = basename(img_path)
            _dirname = basename(dirname(img_path))
            name, ext = splitext(basepath)

            new_name = "%s_%s_%s%s" % (cls, name, _dirname, ext)
            new_path = join(img_dir, new_name)

            if img_idx >= args.n_valid:
                train_paths[new_name] = cls_idx
            else:
                val_paths[new_name] = cls_idx

            if islink(new_path):
                os.unlink(new_path)

            os.symlink(abspath(img_path), new_path)
        cls_idx += 1

    if args.mode == "novel":
        # Novel Class: Random select k images for training, others for testing
        for cls in sorted(os.listdir(join(train_dir, "novel"))):
            cls_dir = join(train_dir, "novel", cls, "train")

            novel_train = random.sample(os.listdir(cls_dir), args.kshot)
            for img_path in os.listdir(cls_dir):
                new_name = "%s_%s" % (cls, img_path)
                new_path = join(img_dir, new_name)

                full_path = join(cls_dir, img_path)
                if img_path in novel_train:
                    if args.oversampling > 1:
                        name, ext = splitext(img_path)
                        for idx in range(args.oversampling):
                            new_name = "%s_%s_%d%s" % (cls, name, idx, ext)
                            new_path = join(img_dir, new_name)

                            if islink(new_path):
                                os.unlink(new_path)
                            os.symlink(abspath(full_path), new_path)
                            train_paths[new_name] = cls_idx
                    else:
                        if islink(new_path):
                            os.unlink(new_path)
                        os.symlink(abspath(full_path), new_path)
                        train_paths[new_name] = cls_idx
                else:
                    val_paths[new_name] = cls_idx
                    if islink(new_path):
                        os.unlink(new_path)
                    os.symlink(abspath(full_path), new_path)
            cls_idx += 1

    train_csv = join(args.output_dir, "train.csv")
    with open(train_csv, 'w') as outf:
        for path, label in train_paths.items():
            outf.write("%s,%s\n" % (path, label))

    val_csv = join(args.output_dir, "val.csv")
    with open(val_csv, 'w') as outf:
        for path, label in val_paths.items():
            outf.write("%s,%s\n" % (path, label))

if __name__ == '__main__':
    main()
