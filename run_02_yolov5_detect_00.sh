#!/bin/bash

CUR_DIR=$PWD

source ./venv/bin/activate
python yolov5/detect.py \
--source input/preprocessed/test_1280/images \
--project output/detect/v56_top3 \
--name test_1280 \
--weights \
${CUR_DIR}/outputs/yolov5_00/weights/best.pt \
${CUR_DIR}/outputs/yolov5_01/weights/best.pt \
${CUR_DIR}/outputs/yolov5_02/weights/best.pt \
${CUR_DIR}/outputs/yolov5_03/weights/best.pt \
${CUR_DIR}/outputs/yolov5_04/weights/best.pt \
${CUR_DIR}/outputs/yolov5_10/weights/best.pt \
${CUR_DIR}/outputs/yolov5_11/weights/best.pt \
${CUR_DIR}/outputs/yolov5_12/weights/best.pt \
${CUR_DIR}/outputs/yolov5_13/weights/best.pt \
${CUR_DIR}/outputs/yolov5_14/weights/best.pt \
--img 832 \
--augment \
--conf-thres 0.05 \
--max-det 3 \
--save-txt \
--save-conf \
--save-crop \
--half