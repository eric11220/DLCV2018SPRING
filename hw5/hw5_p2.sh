#!/bin/bash
set -x
python3 test.py --video_dir $1 --model_path models/trimmed_epoch8-val_acc:0.54-val_loss:1.83.hdf5 --output_path $3/p2_result.txt --part 2 --label_path $2/gt_valid.csv
