#!/usr/bin/env bash

LR=1e-5
alpha=20000  # For the ablation study, use different values
beta=20      # For the ablation study, use different values

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --modifedEnv \
--beta_smooth ${beta} --bias_coef ${alpha} --lr ${LR} --exp_index 0
