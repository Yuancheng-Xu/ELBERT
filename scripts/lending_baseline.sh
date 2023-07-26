#!/usr/bin/env bash

# G-PPO

cd ../lending_experiment
python main.py --train_timesteps 5000000 --algorithm GPPO --exp_index 0


# R-PPO

cd ../lending_experiment
python main.py --train_timesteps 5000000 --algorithm RPPO --zeta_1 2 --exp_index 0


# A-PPO

cd ../lending_experiment
python main.py --train_timesteps 5000000 --algorithm APPO --exp_index 0
