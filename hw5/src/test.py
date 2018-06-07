import argparse
import matplotlib
matplotlib.use('Agg')

import os
import time
from datetime import datetime
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.manifold import TSNE

from data_loader import DataLoader
from dispatcher import Dispatcher
from model import *

PRESAVED_DIR = "../presaved"

# labels should be of shape (n_seqs, seq_len)
def split_labels_to_vids(labels, vid_lens):
    seq_len = labels.shape[1]
    labels = np.reshape(labels, (-1,))

    start = 0
    all_sets = []
    for vid_len in vid_lens:
        if vid_len % seq_len == 0:
            n_skip = vid_len
        else:
            n_skip = (int(vid_len / 20) + 1) * seq_len

        label_set = labels[start:start+vid_len]
        all_sets.append(label_set)
        start += n_skip
    return all_sets

def write_labels_to_file(path, labels):
    with open(path, "w") as outf:
        for label in labels:
            outf.write("%d\n" % label)

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Data folder", default="../HW5_data/TrimmedVideos")
    parser.add_argument("--label_path", help="Ground truth path", default=None)
    parser.add_argument("--model_path", help="Trained model", default="../models/trimmed_06-05:13:26/epoch2-val_acc:0.52-val_loss:1.47.hdf5")
    parser.add_argument("--output_path", help="Output directory or file", default="trimmed_rnn.txt")
    parser.add_argument("--part", help="Which part of homework", default=1, type=int)
    return parser.parse_args()

def main():
    args = parse_input()
    model = load_model(args.model_path)

    # Load data
    data_loader = DataLoader(presaved_dir="../presaved", cut_len=20, cut_step=20)
    if args.part == 1:
        trimmed_valid_conv_feats, valid_labels = \
                data_loader.get_conv_feats("trimmed", video_dir=args.video_dir, label_path=args.label_path, train=False)
        labels = model.predict(trimmed_valid_conv_feats)
        labels = np.argmax(labels, axis=-1)
        write_labels_to_file(args.output_path, labels)
    elif args.part == 2 or args.part == 3:
        if args.part == 2:
            trimmed_valid_conv_feats, valid_labels = \
                    data_loader.get_conv_feats("trimmed", video_dir=args.video_dir, label_path=args.label_path, train=False, raw=True)
        else:
            trimmed_valid_conv_feats, valid_labels, vid_lens, video_cats = \
                    data_loader.get_conv_feats("full", video_dir=args.video_dir, label_path=args.label_path, train=False, raw=True)

        max_seq_len = max([len(conv_feats) for conv_feats in trimmed_valid_conv_feats])
        valid_dispatcher = Dispatcher(trimmed_valid_conv_feats, valid_labels, max_seq_len, shuffle=False)

        all_labels = None
        seqs, _ = valid_dispatcher.next_batch()
        while seqs is not None:
            labels = model.predict(seqs)
            labels = np.argmax(labels, axis=-1)

            all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)
            seqs, _ = valid_dispatcher.next_batch()

        if args.part == 2:
            write_labels_to_file(args.output_path, all_labels)
        else:
            label_sets = split_labels_to_vids(all_labels, vid_lens)
            for label_set, video_cat in zip(label_sets, video_cats):
                path = os.path.join(args.output_path, "%s.txt" % video_cat)
                write_labels_to_file(path, label_set)
    elif args.part == 4:
        avg_valid_conv_feats, valid_labels = \
                data_loader.get_conv_feats("trimmed", video_dir=args.video_dir, label_path=args.label_path, train=False)

        valid_conv_feats, valid_labels = \
                data_loader.get_conv_feats("trimmed", video_dir=args.video_dir, label_path=args.label_path, train=False, raw=True)

        valid_labels = np.argmax(valid_labels, axis=-1)

        feat_extractor = rnn_feat_extractor(model)

        max_seq_len = 267
        dispatcher = Dispatcher(valid_conv_feats, valid_labels, max_seq_len)

        all_feats = None
        seqs, _ = dispatcher.next_batch()
        while seqs is not None:
            feats = feat_extractor([seqs, 0])[0]

            all_feats = feats if all_feats is None else np.concatenate((all_feats, feats), axis=0)
            seqs, _ = dispatcher.next_batch()

        plot_tsne(avg_valid_conv_feats, valid_labels, "../results/conv.jpg")
        plot_tsne(all_feats, valid_labels, "../results/rnn.jpg")

def plot_tsne(feats, labels, path):
    n_class = int(np.max(labels))
    feats_embedded = TSNE(n_components=2).fit_transform(feats)

    plt.clf()
    for i in range(n_class):
        ind = labels == i
        plt.scatter(feats_embedded[ind, 0], feats_embedded[ind, 1])
    plt.savefig(path)

if __name__ == '__main__':
    main()
