import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import to_categorical
from scipy import misc

from reader import *
from model import *

class DataLoader():
    def __init__(self, data_root="HW5_data", presaved_dir="presaved", cut_len=20, cut_step=10, keep_remain=True):
        self.data_root = data_root
        self.model = None
        self.presaved_dir = presaved_dir

        self.cut_len = cut_len
        self.cut_step = cut_step
        self.keep_remain = keep_remain

    def load_trimmed_vids(self, video_dir, label_path, train=False, presaved=None):
        if presaved is not None and os.path.isfile(presaved):
            info = np.load(presaved)
            frame_list, label_list = info['frames'], info['labels']
        else:
            frame_list = []
            label_list = []
    
            info = getVideoList(label_path)
            for name, cat, label in zip(info['Video_name'], info['Video_category'], info['Action_labels']):
                print("Loading %s..." % name)
                frames = readShortVideo(video_dir, cat, name)

                # 'RGB'->'BGR'
                frames = frames[:, :, :, ::-1]
                frame_list.append(frames)
                label_list.append(int(label))
    
            label_list = np.asarray(label_list)
            if presaved is not None:
                os.makedirs(os.path.dirname(presaved),exist_ok=True)
                np.savez(presaved, **{"frames": frame_list, "labels": label_list})

        return frame_list, label_list

    def load_full_labels(self, label_root):
        label_list = []
        for label_path in sorted(os.listdir(label_root)):
            labels = np.array([int(line.strip()) for line in open(os.path.join(label_root, label_path))])
            label_list.append(labels)
        return label_list

    def load_full_frames(self, frame_root):
        frame_list = []
        video_cats = []
        for video_dir in sorted(os.listdir(frame_root)):
            video_cats.append(video_dir)
            video_dir = os.path.join(frame_root, video_dir)

            frames = []
            for frame_path in sorted(os.listdir(video_dir)):
                frame_path = os.path.join(video_dir, frame_path)
                frames.append(misc.imread(frame_path))
            frame_list.append(np.array(frames))
        return frame_list, video_cats

    def load_full_vids(self, video_dir, label_path, train=False, presaved=None):
        if presaved is not None and os.path.isfile(presaved):
            info = np.load(presaved)
            frame_list, label_list = info['frames'], info['labels']
            video_cats = info['video_cats'] if 'video_cats' in info.files else None
        else:
            if label_path is not None:
                label_list = self.load_full_labels(label_path)
            else:
                label_list = None

            frame_list, video_cats = self.load_full_frames(video_dir)
            if presaved is not None:
                os.makedirs(os.path.dirname(presaved),exist_ok=True)
                np.savez(presaved, **{"frames": frame_list, "labels": label_list})

        return frame_list, label_list, video_cats

    def cut_short_seqs(self, frame_list, labels):
        l = self.cut_len
        step = self.cut_step
        keep_remain = self.keep_remain

        cut_frame_list, cut_label_list, vid_lens = [], [], []
        for vid_id, frames in enumerate(frame_list):
            vid_lens.append(len(frames))
            for frame_id in range(0, len(frames), step):
                seq = frames[frame_id:frame_id+l]
                if len(seq) < l and not keep_remain:
                    break
                cut_frame_list.append(seq)

                if labels is not None:
                    seq_label = labels[vid_id][frame_id:frame_id+l]
                    cut_label_list.append(seq_label)

        if labels is not None:
            labels = np.array(cut_label_list)
        else:
            labels = None
        return np.array(cut_frame_list), labels, np.asarray(vid_lens)

    # "raw" indicates whether to perfrom post-processing on conv features
    def get_conv_feats(self, source, video_dir=None, label_path=None, train=True, n_class=11, raw=False):
        if self.presaved_dir is not None:
            if source == "trimmed":
                presaved_path = os.path.join(self.presaved_dir, "trimmed_%s.npz" % ("train" if train else "valid"))
                if raw:
                    presaved_conv_feat_path = os.path.join(self.presaved_dir, "trimmed_%s_conv_feats_raw.npz" % ("train" if train else "valid"))
                else:
                    presaved_conv_feat_path = os.path.join(self.presaved_dir, "trimmed_%s_conv_feats.npz" % ("train" if train else "valid"))
            else:
                presaved_path = os.path.join(self.presaved_dir, "%s.npz" % ("train" if train else "valid"))
                presaved_conv_feat_path = os.path.join(self.presaved_dir, "%s_conv_feats.npz" % ("train" if train else "valid"))
        else:
            presaved_path = None

        if self.presaved_dir is not None and os.path.isfile(presaved_conv_feat_path):
            info = np.load(presaved_conv_feat_path)
            conv_feats, labels = info['conv_feats'], info['labels']
            video_cats = info['video_cats'] if 'video_cats' in info.files else None
        else:
            if source == "trimmed":
                if video_dir is None:
                    video_dir = os.path.join(self.data_root, "TrimmedVideos", "video", "train" if train else "valid")

                if label_path is None:
                    label_path = os.path.join(self.data_root, "TrimmedVideos", "label", "%s.csv" % ("gt_train" if train else "gt_valid"))

                # Load frames
                frame_list, labels = self.load_trimmed_vids(video_dir, label_path, train=train, presaved=presaved_path)
            else:
                if video_dir is None:
                    video_dir = os.path.join(self.data_root, "FullLengthVideos", "videos", "train" if train else "valid")
                frame_list, labels, video_cats = self.load_full_vids(video_dir, label_path, train=train, presaved=presaved_path)

            # Get conv feats after resnet50
            conv_feats = self.predict_images(frame_list, raw=raw)
            if self.presaved_dir is not None:
                np.savez(presaved_conv_feat_path, **{'conv_feats': conv_feats, 'labels': labels})

        if source == "trimmed":
            labels = to_categorical(labels, num_classes=n_class)
            return conv_feats, labels
        else:
            if labels is not None:
                labels = [to_categorical(label_set, num_classes=n_class) for label_set in labels]

            conv_feats, labels, vid_lens = self.cut_short_seqs(conv_feats, labels)
            return conv_feats, labels, vid_lens, video_cats

    def predict_images(self, frame_list, post_process="avg_pool", raw=False, mean=np.asarray([123.68, 116.779, 103.939])):
        if self.model is None:
            self.model = load_model("pretrained-resnet50.hdf5")

        conv_feats = []
        for idx, frames in enumerate(frame_list):
            print("Forwarding %d set of frames thru network..." % idx)
            conv_feat = self.model.predict([frames, np.tile(mean, frames.shape[:-1] + (1,))])
            conv_feat = conv_feat[:, 0, 0, :]
    
            if not raw:
                if post_process == "avg_pool":
                    conv_feat = np.mean(conv_feat, axis=0)
            conv_feats.append(conv_feat)
    
        conv_feats = np.asarray(conv_feats)
        return conv_feats
