#!/bin/bash

source ./venv/bin/activate
python run_tta.py \
exec_time=2021-1031-0000 \
fold=3 \
exp=exp00_effb4_380 \
exp.model.pretrained=False