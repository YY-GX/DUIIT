#!/bin/sh
pip install -r requirements.txt
pip install --upgrade torch torchvision  
# conda env create -f ./checkpoints/latest_pet/env.yml
# source activate
# conda info --envs
# conda activate base
# python test.py --dataroot ../../datasets/mr_ct_raw_dataset/pet/trainA --name latest_pet --model test --no_dropout --results_dir ../../datasets/mr_ct_trans_dataset/pet_trans_gpu  --num_test 12780
python test.py  --dataroot ../../datasets/final_mr_ct/pet_mri/trainA  --name pet_mri --model test --no_dropout --results_dir ../../datasets/final_mr_ct/pet_mri_fake/ --num_test 16234