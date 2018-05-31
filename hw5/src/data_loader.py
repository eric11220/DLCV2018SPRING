import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from reader import *
from model import *

class DataLoader():
    def __init__(self, data_root="../HW5_data", presaved_dir="../presaved"):
        self.data_root = data_root
        self.model = None
        self.presaved_dir = presaved_dir

    def load_trimmed_vids(self, data_dir, train=False, presaved=None):
        if presaved is not None and os.path.isfile(presaved):
            info = np.load(presaved)
            frame_list, label_list = info['frames'], info['labels']
        else:
            label_path = os.path.join(data_dir, "label", "%s.csv" % ("gt_train" if train else "gt_valid"))
            video_root = os.path.join(data_dir, "video", "train" if train else "valid")
                
            frame_list = []
            label_list = []
    
            info = getVideoList(label_path)
            for name, cat, label in zip(info['Video_name'], info['Video_category'], info['Action_labels']):
                print("Loading %s..." % name)
                frames = readShortVideo(video_root, cat, name)

                # 'RGB'->'BGR'
                frames = frames[:, :, :, ::-1]
                frame_list.append(frames)
                label_list.append(int(label))
    
            label_list = np.asarray(label_list)
            if presaved is not None:
                np.savez(presaved, **{"frames": frame_list, "labels": label_list})

        return frame_list, label_list

    def load_full_labels(self, label_root):
        label_list = []
        for label_path in os.listdir(label_root):
            labels = [int(line.strip()) for line in open(os.path.join(label_root, label_path))]
            print(labels)
            input("check labels of a single video")
            label_list.append(labels)
        return label_list

    def load_full_vids(self, data_dir, train=False, presaved=None):
        if presaved is not None and os.path.isfile(presaved):
            info = np.load(presaved)
            frame_list, label_list = info['frames'], info['labels']
        else:
            label_root = os.path.join(data_dir, "labels", "train" if train else "valid")
            video_root = os.path.join(data_dir, "videos", "train" if train else "valid")

            labels = self.load_full_labels(label_root)
            frame_list = self.load_full_frames(video_root)

    # "raw" indicates whether to perfrom post-processing on conv features
    def get_conv_feats(self, source, train=True, n_class=11, raw=False):
        if source == "trimmed":
            if raw:
                presaved_conv_feat_path = os.path.join(self.presaved_dir, "trimmed_%s_conv_feats_raw.npz" % ("train" if train else "valid"))
            else:
                presaved_conv_feat_path = os.path.join(self.presaved_dir, "trimmed_%s_conv_feats.npz" % ("train" if train else "valid"))
        else:
            presaved_conv_feat_path = os.path.join(self.presaved_dir, "%s_conv_feats.npz" % ("train" if train else "valid"))

        if os.path.isfile(presaved_conv_feat_path):
            info = np.load(presaved_conv_feat_path)
            conv_feats, labels = info['conv_feats'], info['labels']
        else:
            if source == "trimmed":
                # Load frames
                data_dir = os.path.join(self.data_root, "TrimmedVideos")
                presaved_path = os.path.join(self.presaved_dir, "trimmed_%s.npz" % ("train" if train else "valid"))
                frame_list, labels = self.load_trimmed_vids(data_dir, train=train, presaved=presaved_path)
            else:
                data_dir = os.path.join(self.data_root, "TrimmedVideos")
                presaved_path = os.path.join(self.presaved_dir, "%s.npz" % ("train" if train else "valid"))
                frame_list, labels = self.load_full_frames(data_dir, train=train, presaved=presaved_path)

            # Get conv feats after resnet50
            conv_feats = self.predict_images(frame_list, raw=raw)
            np.savez(presaved_conv_feat_path, **{'conv_feats': conv_feats, 'labels': labels})

        labels = to_categorical(labels, num_classes=n_class)
        return conv_feats, labels

    def predict_images(self, frame_list, post_process="avg_pool", raw=False, mean=np.asarray([123.68, 116.779, 103.939])):
        if self.model is None:
            self.model = resnet50()

        conv_feats = []
        for idx, frames in enumerate(frame_list):
            conv_feat = self.model.predict([frames, np.tile(mean, frames.shape[:-1] + (1,))])
            conv_feat = conv_feat[:, 0, 0, :]
    
            if not raw:
                if post_process == "avg_pool":
                    conv_feat = np.mean(conv_feat, axis=0)
            conv_feats.append(conv_feat)
    
        conv_feats = np.asarray(conv_feats)
        return conv_feats
