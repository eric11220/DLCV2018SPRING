#!/bin/bash
set -x
python3 test.py --video_dir $1 --model_path models/full_epoch1-val_acc:0.59-val_loss:1.26.hdf5  --output_path $2 --part 3
