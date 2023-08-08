#!/usr/bin/env bash

LR=1e-6
alpha=20   # For the ablation study, use different values

cd ../infectious_experiment
python main.py --train_timesteps 10000000 --bias_coef ${alpha} --exp_index 0
