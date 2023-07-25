#!/usr/bin/env bash

LR=1e-5

# G-PPO

cd ../lending_experiment
python main.py --train_timesteps 5000000 --algorithm GPPO --lr ${LR} --exp_index 0


# R-PPO

cd ../lending_experiment
python main.py --train_timesteps 5000000 --algorithm RPPO --zeta_1 2 --lr ${LR} --exp_index 0


# A-PPO

cd ../lending_experiment
python main.py --train_timesteps 5000000 --algorithm APPO --lr ${LR} --exp_index 0
