#!/usr/bin/env bash

LR=1e-5

# G-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --modifedEnv \
--algorithm GPPO --lr ${LR} --exp_index 0


# R-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --modifedEnv \
--algorithm RPPO --zeta_2 20 --lr ${LR} --exp_index 0


# A-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --modifedEnv \
--algorithm APPO --lr ${LR} --exp_index 0
