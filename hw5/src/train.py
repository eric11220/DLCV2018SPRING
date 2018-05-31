import argparse
import time
from data_loader import DataLoader
from keras.callbacks import ModelCheckpoint
from model import *

PRESAVED_DIR = "../presaved"

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Data folder", default="../HW5_data/")
    parser.add_argument("--model_dir", help="Trained model directory", default="../models")
    parser.add_argument("--n_epoch", help="Number of epochs", default=50)
    return parser.parse_args()

def debug_read_data():
    data_loader = DataLoader()
    trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True) 
    trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False) 
    return trimmed_train_conv_feats, train_labels, trimmed_valid_conv_feats, valid_labels

def main(trimmed_train_conv_feats, train_labels, trimmed_valid_conv_feats, valid_labels):
    args = parse_input()
    os.makedirs(args.model_dir, exist_ok=True)

    data_loader = DataLoader()
    trimmed_train_conv_feats, train_labels = data_loader.get_conv_feats("trimmed", train=True) 
    trimmed_valid_conv_feats, valid_labels = data_loader.get_conv_feats("trimmed", train=False) 

    model = action_fc_model()
    model_ckpoint = ModelCheckpoint(os.path.join(args.model_dir, "conv-{epoch:02d}-{val_accu:.2f}-{val_loss:.2f}.hdf5"),
                                    monitor='val_accu')
    model.fit(trimmed_train_conv_feats, train_labels,
              epochs=args.n_epoch,
              validation_data=(trimmed_valid_conv_feats, valid_labels),
              callbacks=[model_ckpoint])

if __name__ == '__main__':
    main()
