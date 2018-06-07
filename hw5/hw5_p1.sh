#!/bin/bash
set -x
python3 test.py --video_dir $1 --model_path models/fc_epoch34-0.50-2.64.hdf5 --output_path $3/p1_valid.txt --part 1 --label_path $2
