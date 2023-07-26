#!/usr/bin/env bash

alpha=200000   # For the ablation study, use different values

cd ../lending_experiment
python main.py --train_timesteps 5000000 --bias_coef ${alpha} --exp_index 0