#!/usr/bin/env bash

alpha=20000  # For the ablation study, use different values
beta=20      # For the ablation study, use different values

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --harderEnv --beta_smooth ${beta} --bias_coef ${alpha} --exp_index 0
