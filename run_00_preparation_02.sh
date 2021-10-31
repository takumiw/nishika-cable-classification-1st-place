#!/bin/bash

CUR_DIR=$PWD

git clone https://github.com/ultralytics/yolov5
cp ./tmp/cable_cv*.yaml ./yolov5/data/

sed -ie "s|.\/input|${CUR_DIR}\/input|g" ./input/yolo/train_cv*.txt
sed -ie "s|.\/input|${CUR_DIR}\/input|g" ./input/yolo/valid_cv*.txt