#!/usr/bin/env bash

LR=1e-6

# G-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 10000000 --algorithm GPPO --lr ${LR} --exp_index 0


# R-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 10000000 --algorithm RPPO --zeta_2 20 --lr ${LR} --exp_index 0


# A-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 10000000 --algorithm APPO --lr ${LR} --exp_index 0
