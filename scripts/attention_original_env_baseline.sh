#!/usr/bin/env bash

LR=1e-6

# G-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 20000000 --algorithm GPPO --lr ${LR} --exp_index 0


# R-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 20000000 --algorithm RPPO --zeta_2 10 --lr ${LR} --exp_index 0


# A-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 20000000 --algorithm APPO --lr ${LR} --exp_index 0
