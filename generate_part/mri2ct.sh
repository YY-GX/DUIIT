#!/bin/sh
pip install -r requirements.txt &&
pip install --upgrade torch torchvision   &&

python train.py --dataroot ./../datasets/mri/  --name mri --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/mri --display_id 0 --gpu_ids 0,1,2,3  --log_dir logs/mri/ --save_epoch_freq 10 --arch resnet18  --batch_size 16  --rgr_lr 0.001 --fc_input 512 

