#!/bin/bash

cd data
python restructure.py --mode novel --n-valid 0 --output-dir ../src/Random-Erasing/ten_shot

cd src/Random-Erasing/
python3 test.py checkpoint/model_best.pth.tar --mode test --out-file 1.csv  --kshot 1  --gather
python3 test.py checkpoint/model_best.pth.tar --mode test --out-file 5.csv  --kshot 5  --gather
python3 test.py checkpoint/model_best.pth.tar --mode test --out-file 10.csv --kshot 10 --gather
