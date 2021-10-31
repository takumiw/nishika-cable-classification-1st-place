#!/bin/bash

source ./venv/bin/activate
python run_tta.py \
exec_time=2021-1031-0500 \
fold=1 \
exp=exp05_resnet101d_256 \
exp.model.pretrained=False