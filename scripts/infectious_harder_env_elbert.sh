#!/usr/bin/env bash

alpha=50   # For the ablation study, use different values

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --harderEnv --bias_coef ${alpha} --exp_index 0