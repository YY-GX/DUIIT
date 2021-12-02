#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision 

python pet_ct.py -a resnet18 --logpath logs/log_pet_ct/ --resumedir checkpoints/checkpoint_pet_ct/ --filename outcome_log_pet_ct.csv --augement pure --train-dir './../datasets/real_ct/train/' --val-dir './../datasets/real_ct/val/' --test-dir './../datasets/real_ct/test/' --augedir './../datasets/fake_pet2ct/' --epochs 100 --times 1  -b 64 --dropout 0 --lr 1e-4 --wd 0 datasets 