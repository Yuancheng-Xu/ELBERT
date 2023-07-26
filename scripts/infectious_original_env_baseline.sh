#!/usr/bin/env bash

# G-PPO

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --algorithm GPPO --exp_index 0

# R-PPO

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --algorithm RPPO --zeta_1 0.1 --exp_index 0


# A-PPO

cd ../infectious_experiment
python main.py --train_timesteps 5000000 --algorithm APPO --exp_index 0
