sleep 2h &&
python train.py --dataroot ../../datasets/half_mr_ct_raw_dataset/  --name 2500_pretrain --model cycle_gan --num_threads 8 --checkpoints_dir checkpoints/2500_pretrain --display_id 0 --gpu_ids 0,1,2,3  --log_dir logs/2500_pretrain/ --save_epoch_freq 1 --arch resnet18  --batch_size 15  --rgr_lr 0.001 --fc_input 512 
