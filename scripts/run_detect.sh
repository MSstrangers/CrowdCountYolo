#!/bin/bash

python detect.py \
--weights crowdhuman_yolov5m.pt  \
--source crowd_sample1_trimmed.mp4 \
--heads \
--view-img \
--conf-thres 0.5 \
--save

