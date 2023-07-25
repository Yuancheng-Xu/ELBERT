#!/usr/bin/env bash

LR=1e-5
alpha=50   # For the ablation study, use different values

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --modifedEnv --bias_coef ${alpha} --lr ${LR} --exp_index 0