#!/bin/sh
pip install -r requirements.txt &&
pip install --upgrade torch torchvision   &&

python train.py --dataroot ./../datasets/pet/  --name pet --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/pet --display_id 0 --gpu_ids 0,1,2,3  --log_dir logs/pet/ --save_epoch_freq 10 --arch resnet18  --batch_size 16  --rgr_lr 0.001 --fc_input 512 

