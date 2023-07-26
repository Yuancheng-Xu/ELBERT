#!/usr/bin/env bash

LR=1e-6
alpha=30     # For the ablation study, use different values
beta=20      # For the ablation study, use different values

cd ../attention_allocation_experiment
python main.py --train_timesteps 10000000 --beta_smooth ${beta} --bias_coef ${alpha} --lr ${LR} --exp_index 0