import argparse
import os
import time
from datetime import datetime
from keras.callbacks import ModelCheckpoint

from data_loader import DataLoader
from dispatcher import Dispatcher
from model import *

PRESAVED_DIR = "../presaved"

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Data folder", default="../HW5_data/")
    parser.add_argument("--model_dir", help="Trained model directory", default="../models")
    parser.add_argument("--part", help="Which part of homework", default=2)
    parser.add_argument("--n_epoch", help="Number of epochs", default=50)
    return parser.parse_args()

def debug_read_data():
    data_loader = DataLoader(presaved_dir="../presaved")
    trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True, raw=True)
    trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False, raw=True)
    return trimmed_train_conv_feats, train_labels, trimmed_valid_conv_feats, valid_labels

def main(trimmed_train_conv_feats, train_labels, trimmed_valid_conv_feats, valid_labels):
    args = parse_input()

    model_dir = os.path.join(args.model_dir, datetime.now().strftime("%m-%d:%H:%M"))
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    data_loader = DataLoader(presaved_dir="../presaved")
    if args.part == 1:
        trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True)
        trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False)

        model = action_fc_model()
        model_ckpoint = ModelCheckpoint(os.path.join(model_dir, "conv-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5"),
                                        monitor='val_acc',
                                        save_best_only=True)
        model.fit(trimmed_train_conv_feats, train_labels,
                  epochs=args.n_epoch,
                  validation_data=(trimmed_valid_conv_feats, valid_labels),
                  callbacks=[model_ckpoint])

    elif args.part == 2:
        trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True, raw=True)
        trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False, raw=True)

        train_max_seq_len = max([len(conv_feats) for conv_feats in trimmed_train_conv_feats])
        valid_max_seq_len = max([len(conv_feats) for conv_feats in trimmed_valid_conv_feats])
        max_seq_len = max(train_max_seq_len, valid_max_seq_len)
        model = rnn((max_seq_len, trimmed_train_conv_feats[0].shape[1]))

        cnt = 0
        train_dispatcher = Dispatcher(trimmed_train_conv_feats, train_labels, max_seq_len)
        valid_dispatcher = Dispatcher(trimmed_valid_conv_feats, valid_labels, max_seq_len)
        for epoch_id in range(args.n_epoch):
            print("Epoch %d..." % epoch_id)

            losses, accus = [], []
            seqs, labels = train_dispatcher.next_batch()
            while seqs is not None:
                loss, accu = model.train_on_batch(seqs, labels)
                losses.append(loss);
                accus.append(accu)

                seqs, labels = train_dispatcher.next_batch()
                if seqs is not None:
                    cnt += len(seqs)
                    print("Processed %d data..." % cnt)
                else:
                    valid_stats= None
                    valid_seqs, valid_labels = valid_dispatcher.next_batch()
                    while valid_seqs is not None:
                        stat = model.evaluate(valid_seqs, valid_labels, verbose=0)
                        valid_stats = stat if valid_stats is None else np.vstack((valid_stats, stat))
                        valid_seqs, valid_labels = valid_dispatcher.next_batch()

                    print(valid_stats)
                    print("Training loss: %.2f, accurcy: %.2f" % (np.mean(losses), np.mean(accus)))
                    print("Validation loss: %.2f, accurcy: %.2f"
                                % (np.mean(valid_stats[:, 0]), np.mean(valid_stats[:, 1])))

if __name__ == '__main__':
    main()
