#!/bin/bash
pip install -r requirements.txt

python test.py --dataroot ./../datasets/real_pet/ --name latest_pet --model test --no_dropout --results_dir ./../datasets/fake_pet2ct/  --num_test 12780
