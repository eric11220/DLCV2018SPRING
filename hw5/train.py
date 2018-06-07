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
    parser.add_argument("--part", help="Which part of homework", default=3)
    parser.add_argument("--n_epoch", help="Number of epochs", default=50)
    return parser.parse_args()

def debug_read_data():
    data_loader = DataLoader(presaved_dir="../presaved", keep_remain=False)
    #trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True)
    #trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False)
    #trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True, raw=True)
    #trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False, raw=True)
    trimmed_train_conv_feats, train_labels, _, _ = data_loader.get_conv_feats("full", train=True, raw=True)
    trimmed_valid_conv_feats, valid_labels, _, _ = data_loader.get_conv_feats("full", train=False, raw=True)
    return trimmed_train_conv_feats, train_labels, trimmed_valid_conv_feats, valid_labels

#def main(trimmed_train_conv_feats, train_labels, trimmed_valid_conv_feats, valid_labels):
def main():
    args = parse_input()

    if args.part == 1:
        model_dir = os.path.join(args.model_dir, "conv_%s" % datetime.now().strftime("%m-%d:%H:%M"))
    elif args.part == 2:
        model_dir = os.path.join(args.model_dir, "trimmed_%s" % datetime.now().strftime("%m-%d:%H:%M"))
    else:
        model_dir = os.path.join(args.model_dir, "full_%s" % datetime.now().strftime("%m-%d:%H:%M"))
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    data_loader = DataLoader(presaved_dir="../presaved", keep_remain=False)
    if args.part == 1:
        trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True)
        trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False)

        model = action_fc_model()
        model_ckpoint = ModelCheckpoint(os.path.join(model_dir, "{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5"),
                                        monitor='val_acc',
                                        save_best_only=True)
        hist = model.fit(trimmed_train_conv_feats, train_labels,
                         epochs=args.n_epoch,
                         validation_data=(trimmed_valid_conv_feats, valid_labels),
                         callbacks=[model_ckpoint])

        curve_path = os.path.join(model_dir, "learning_curve.jpg")
        hist = hist.history
        plot_training_curve(hist['acc'], hist['loss'], hist['val_acc'], hist['val_loss'], curve_path)

    else:
        if args.part == 2:
            train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True, raw=True)
            valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False, raw=True)
        else:
            train_conv_feats, train_labels, _, _ = data_loader.get_conv_feats("full", train=True, raw=True)
            valid_conv_feats, valid_labels, _, _ = data_loader.get_conv_feats("full", train=False, raw=True)

        train_max_seq_len = max([len(conv_feats) for conv_feats in train_conv_feats])
        valid_max_seq_len = max([len(conv_feats) for conv_feats in valid_conv_feats])
        max_seq_len = max(train_max_seq_len, valid_max_seq_len)

        if args.part == 2:
            model = rnn((max_seq_len, train_conv_feats[0].shape[1]))
        else:
            model = rnn((max_seq_len, train_conv_feats[0].shape[1]), return_sequences=True)

        best_accu = 0.
        train_dispatcher = Dispatcher(train_conv_feats, train_labels, max_seq_len)
        valid_dispatcher = Dispatcher(valid_conv_feats, valid_labels, max_seq_len)

        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        for epoch_id in range(args.n_epoch):
            print("Epoch %d..." % epoch_id)

            cnt = 0
            losses, accus = [], []
            seqs, labels = train_dispatcher.next_batch()
            while seqs is not None:
                loss, accu = model.train_on_batch(seqs, labels)
                losses.append(loss);
                accus.append(accu)

                seqs, labels = train_dispatcher.next_batch()
                if seqs is not None:
                    cnt += len(seqs)
                else:
                    valid_stats= None
                    valid_seqs, valid_labels = valid_dispatcher.next_batch()
                    while valid_seqs is not None:
                        stat = model.evaluate(valid_seqs, valid_labels, verbose=0)
                        valid_stats = stat if valid_stats is None else np.vstack((valid_stats, stat))
                        valid_seqs, valid_labels = valid_dispatcher.next_batch()

                    valid_loss = np.mean(valid_stats[:, 0])
                    valid_accu = np.mean(valid_stats[:, 1])
                    if valid_accu > best_accu:
                        best_accu = valid_accu
                        model_path = os.path.join(model_dir, "epoch%d-val_acc:%.2f-val_loss:%.2f.hdf5"
                                                             % (epoch_id, valid_accu, valid_loss))
                        model.save(model_path)

                    train_loss.append(np.mean(losses))
                    train_acc.append(np.mean(accus))
                    val_loss.append(valid_loss)
                    val_acc.append(valid_accu)
                    print("Training loss: %.2f, accurcy: %.2f" % (np.mean(losses), np.mean(accus)))
                    print("Validation loss: %.2f, accurcy: %.2f" % (valid_loss, valid_accu))

        curve_path = os.path.join(model_dir, "learning_curve.jpg")
        plot_training_curve(train_acc, train_loss, val_acc, val_loss, curve_path)

if __name__ == '__main__':
    main()
