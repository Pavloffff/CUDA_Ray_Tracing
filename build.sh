#! /bin/bash

arg=$1
make
./kp $arg < t1.txt
python3 converter.py 300
