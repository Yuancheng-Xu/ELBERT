#!/usr/bin/env bash

LR=1e-5

# G-PPO

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --modifedEnv --algorithm GPPO --lr ${LR} --exp_index 0


# R-PPO

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --modifedEnv --algorithm RPPO --zeta_1 20 --lr ${LR} --exp_index 0


# A-PPO

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --modifedEnv --algorithm APPO --lr ${LR} --exp_index 0
