#!/usr/bin/env bash

trap 'kill 0' SIGINT

CUDA_VISIBLE_DEVICES=0 python main_fair.py --modifedEnv --bias_coef 0.0 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_0 &
CUDA_VISIBLE_DEVICES=1 python main_fair.py --modifedEnv --bias_coef 2000.0 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_2000 &
CUDA_VISIBLE_DEVICES=2 python main_fair.py --modifedEnv --bias_coef 20.0 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_20 &
CUDA_VISIBLE_DEVICES=3 python main_fair.py --modifedEnv --bias_coef 50.0 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_50 &
CUDA_VISIBLE_DEVICES=4 python main_fair.py --modifedEnv --bias_coef 100 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_100 &
CUDA_VISIBLE_DEVICES=5 python main_fair.py --modifedEnv --bias_coef 200 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_200 &
CUDA_VISIBLE_DEVICES=6 python main_fair.py --modifedEnv --bias_coef 400 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_400 &
CUDA_VISIBLE_DEVICES=7 python main_fair.py --modifedEnv --bias_coef 1000 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_1000 










# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 8.0 --exp_path b_8 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 10.0 --exp_path b_10 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --bias_coef 20.0 --exp_path b_20 
# the last command must not include the last &