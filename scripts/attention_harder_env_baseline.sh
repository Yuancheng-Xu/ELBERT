#!/usr/bin/env bash

# G-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --harderEnv --algorithm GPPO --exp_index 0


# R-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --harderEnv --algorithm RPPO --zeta_2 20 --exp_index 0


# A-PPO

cd ../attention_allocation_experiment
python main.py --train_timesteps 5000000 --zeta_0 0 --harderEnv --algorithm APPO --exp_index 0
