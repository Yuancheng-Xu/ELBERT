#!/usr/bin/env bash

trap 'kill 0' SIGINT

# GPPO and APPO: specifying --algorithm GPPO/APPO is enough
# RPPO: also need to specify --zeta_1
# ours: specify --bias_coef (no need for beta_smooth since only 2 groups)

### modified env

# ours
# LR=1e-5
# coef_list=(10 15 20 30)

# CUDA_VISIBLE_DEVICES=0 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --lr $LR --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 

# baselines
# CUDA_VISIBLE_DEVICES=0 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_2 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 0.1 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 0.1 --exp_path_extra _s_1 &
# CUDA_VISIBLE_DEVICES=7 python main.py --modifedEnv --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 0.1 --exp_path_extra _s_2


### Eric's original env

# ours
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --bias_coef 0 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --bias_coef 0 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --bias_coef 1 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --bias_coef 1 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --exp_path_env original \
# --bias_coef 5 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --exp_path_env original \
# --bias_coef 5 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --exp_path_env original \
# --bias_coef 20 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --exp_path_env original \
# --bias_coef 20 --exp_path_extra _s_1 

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
# CUDA_VISIBLE_DEVICES=5 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_1 0.1 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_1 0.1 --exp_path_extra _s_1 &
# CUDA_VISIBLE_DEVICES=7 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_1 0.1 --exp_path_extra _s_2