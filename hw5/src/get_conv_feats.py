import argparse
from data_loader import *
from model import *

def get_conv_feats(model, frame_list, source, post_process="avg_pool"):
    conv_feats = []
    for idx, frames in enumerate(frame_list):
        conv_feat = model.predict(frames)
        conv_feat = conv_feat[:, 0, 0, :]

        if post_process == "avg_pool":
            conv_feat = np.mean(conv_feat, axis=0)
        conv_feats.append(conv_feat)

    conv_feats = np.asarray(conv_feats)
    return conv_feats

def main(frame_list, labels):
    args = parse_input()
    presaved_t_train_conv_feat = os.path.join(PRESAVED_DIR, "trimmed_train_conv_feats.npy")

    if not os.path.isfile(presaved_t_train_conv_feat):
        trimmed_dir = os.path.join(args.data_dir, "TrimmedVideos")
        frame_list, labels = load_trimmed_vids(trimmed_dir, train=True, presaved=os.path.join(PRESAVED_DIR, "trimmed_train.npz"))

        model = resnet50()
        conv_feats = get_conv_feats(model, frame_list)
        np.save(presaved_t_train_conv_feat, conv_feats)

    conv_feats = get_conv_feats(model, frame_list)

if __name__ == '__main__':
    main()
