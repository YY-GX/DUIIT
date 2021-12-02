#!/bin/bash
pip install -r requirements.txt

python test.py --dataroot ./../datasets/real_mri/ --name latest_mri --model test --no_dropout --results_dir ./datasets/fake_mri2ct/  --num_test 3542
