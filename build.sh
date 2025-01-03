#! /bin/bash

cnt=$1
make
./kp $cnt
python3 converter.py $cnt
