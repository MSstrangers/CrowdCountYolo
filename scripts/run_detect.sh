#!/bin/bash

python detect.py \
--weights crowdhuman_yolov5m.pt  \
--source sample_crowd.mp4 \
--conf-thres 0.5 \
--save

# --view-img \

