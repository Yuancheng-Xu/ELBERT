#!/usr/bin/env bash

trap 'kill 0' SIGINT

# GPPO and APPO: specifying --algorithm GPPO/APPO is enough
# RPPO: also need to specify --zeta_2
# ours: specify --beta_smooth and --bias_coef 


### modified env
# ours
# LR=1e-4
# BETA_SMOOTH=5
# # coef_list=(0 4000 6000 8000)
# coef_list=(200 500 1000 2000)

# CUDA_VISIBLE_DEVICES=0 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 

# baselines
# CUDA_VISIBLE_DEVICES=0 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_2 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_3 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _s_1 


### Eric's original env

# ours

# LR=1e-4
# BETA_SMOOTH=5
# coef_list=(0 10 20 50)

# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 

# baselines
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --algorithm GPPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --algorithm GPPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --algorithm APPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --algorithm APPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --exp_path_env original \
# --algorithm APPO --exp_path_extra _s_2 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _s_1 &
# CUDA_VISIBLE_DEVICES=5 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _s_2 