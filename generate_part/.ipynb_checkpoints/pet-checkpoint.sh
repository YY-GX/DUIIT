#!/bin/sh
pip install -r requirements.txt &&
pip install --upgrade torch torchvision   &&
# pip install requirements.txt &&
python train.py --dataroot ../../datasets/mr_ct_raw_dataset/pet/  --name pet --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/pet_4gpu --display_id 0 --gpu_ids 0,1,2,3  --log_dir logs/pet_4gpu/ --save_epoch_freq 10 --arch resnet18  --batch_size 16  --rgr_lr 0.001 --fc_input 512 
# python train.py --dataroot ../../datasets/mr_ct_raw_dataset/pet/  --name pet --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/pet --display_id 0 --gpu_ids 0  --log_dir logs/pet/ --save_epoch_freq 10 --arch resnet18  --batch_size 3  --rgr_lr 0.001 --fc_input 512 
