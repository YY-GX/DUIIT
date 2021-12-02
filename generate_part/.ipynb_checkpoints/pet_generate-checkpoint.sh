#!/bin/bash
pip install -r requirements.txt
conda env create -f ./checkpoints/latest_pet/env.yml
source activate
conda activate yy2
python test.py --dataroot ../../datasets/mr_ct_raw_dataset/pet/trainA --name latest_pet --model test --no_dropout --results_dir ../../datasets/mr_ct_trans_dataset/pet_trans_gpu  --num_test 12780
