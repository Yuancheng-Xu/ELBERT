#!/usr/bin/env bash

trap 'kill 0' SIGINT

# CUDA_VISIBLE_DEVICES=0 python main_fair.py --bias_coef 0.0 --exp_path b_0 &
# CUDA_VISIBLE_DEVICES=0 python main_fair.py --bias_coef 50 --exp_path b_50 

# CUDA_VISIBLE_DEVICES=0 python main_fair.py --bias_coef 0.1 --exp_path b_0.1 &
# CUDA_VISIBLE_DEVICES=0 python main_fair.py --bias_coef 0.5 --exp_path b_0.5 &
# CUDA_VISIBLE_DEVICES=0 python main_fair.py --bias_coef 1.0 --exp_path b_1 &
# CUDA_VISIBLE_DEVICES=0 python main_fair.py --bias_coef 2.0 --exp_path b_2 

# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 4.0 --exp_path b_4 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 8.0 --exp_path b_8 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 10.0 --exp_path b_10 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 20.0 --exp_path b_20 








# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 8.0 --exp_path b_8 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 10.0 --exp_path b_10 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 20.0 --exp_path b_20 
# the last command must not include the last &