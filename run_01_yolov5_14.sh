#!/bin/bash

source ./venv/bin/activate
python src/preprocess/prepare_yolo_labels.py Lightning_T_all
python yolov5/train.py \
--img 640  \
--batch 16  \
--epochs 300  \
--label-smoothing 0.1 \
--data cable_cv4.yaml  \
--weights yolov5x.pt \
--project outputs  \
--name yolov5_14