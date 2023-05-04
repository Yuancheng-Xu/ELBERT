#!/usr/bin/env bash

trap 'kill 0' SIGINT

# CUDA_VISIBLE_DEVICES=0 python main_fair.py --modifedEnv --bias_coef 0 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_0 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --modifedEnv --bias_coef 0.1 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_0.1 &
# CUDA_VISIBLE_DEVICES=2 python main_fair.py --modifedEnv --bias_coef 0.5 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_0.5 &
# CUDA_VISIBLE_DEVICES=3 python main_fair.py --modifedEnv --bias_coef 1 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_1 &
# CUDA_VISIBLE_DEVICES=4 python main_fair.py --modifedEnv --bias_coef 2 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_2 &
# CUDA_VISIBLE_DEVICES=5 python main_fair.py --modifedEnv --bias_coef 5 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_5 &
# CUDA_VISIBLE_DEVICES=6 python main_fair.py --modifedEnv --bias_coef 10 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_10 &
# CUDA_VISIBLE_DEVICES=7 python main_fair.py --modifedEnv --bias_coef 20 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_20 &
# CUDA_VISIBLE_DEVICES=1 python main_fair.py --modifedEnv --bias_coef 15 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_15 &
# CUDA_VISIBLE_DEVICES=2 python main_fair.py --modifedEnv --bias_coef 25 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_25 &
# CUDA_VISIBLE_DEVICES=3 python main_fair.py --modifedEnv --bias_coef 40 --exp_path Chenghao_env_1/lr_1e-6_samples_5e6/b_40 



CUDA_VISIBLE_DEVICES=0 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 0 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_0 &
CUDA_VISIBLE_DEVICES=1 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 0.1 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_0.1 &
CUDA_VISIBLE_DEVICES=2 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 0.5 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_0.5 &
CUDA_VISIBLE_DEVICES=3 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 1 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_1 &
CUDA_VISIBLE_DEVICES=4 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 2 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_2 &
CUDA_VISIBLE_DEVICES=5 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 5 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_5 &
CUDA_VISIBLE_DEVICES=6 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 10 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_10 &
CUDA_VISIBLE_DEVICES=7 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 20 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_20 &
CUDA_VISIBLE_DEVICES=1 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 15 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_15 &
CUDA_VISIBLE_DEVICES=2 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 25 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_25 &
CUDA_VISIBLE_DEVICES=3 python main_fair.py --modifedEnv --lr 1e-5 --bias_coef 40 --exp_path Chenghao_env_1/lr_1e-5/samples_5e6/b_40 


