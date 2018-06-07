import numpy as np
from keras.preprocessing.sequence import pad_sequences

class Dispatcher():
    def __init__(self, conv_feats_list, labels=None, max_seq_len=20, batch_size=16, shuffle=True):
        self.batch_size = batch_size
        self.conv_feats_list = conv_feats_list
        self.labels = labels
        self.max_seq_len = max_seq_len

        self.do_shuffle = shuffle
        self.prepare()

    def next_batch(self):
        if self.start >= len(self.conv_feats_list):
            self.start = 0
            if self.do_shuffle:
                self.shuffle()
            return None, None
        else:
            end = self.start + self.batch_size
            this_ind = self.ind[self.start:end]
            self.start += self.batch_size

            conv_feats = pad_sequences(self.conv_feats_list[this_ind], maxlen=self.max_seq_len)
            if self.labels is not None:
                labels = self.labels[this_ind]
            else:
                labels = None
            return conv_feats, labels

    def prepare(self):
        if self.do_shuffle:
            self.ind = np.random.permutation(len(self.conv_feats_list))
        else:
            self.ind = np.arange(len(self.conv_feats_list))
        self.start = 0

    def shuffle(self):
        np.random.shuffle(self.ind)
