#!/bin/bash

rm ./input/additional/*_trim.jpg
mv ./input/train/*.jpg ./input/train_all
mv ./input/additional/*.jpg ./input/train_all