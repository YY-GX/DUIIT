#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision

python mr_ct.py -a resnet50 --logpath logs/log_mr_ct/ --resumedir checkpoints/checkpoint_mr_ct/ --filename outcome_log_mr_ct.csv --augement 'toge' --train-dir './../datasets/real_ct/train/' --val-dir './../datasets/real_ct/val/' --test-dir './../datasets/real_ct/test/' --augedir './../datasets/fake_mr2ct/' --epochs 100 --times 1  -b 64 --dropout 0 --lr 1e-4  datasets 