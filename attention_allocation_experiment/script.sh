#!/usr/bin/env bash

trap 'kill 0' SIGINT

# GPPO and APPO: specifying --algorithm GPPO/APPO is enough
# RPPO: also need to specify --zeta_2
# ours: specify --beta_smooth and --bias_coef 

##### modified env & new policy evaluation

# Chenghao
# coef needs to include: 0 5 20 50 100 200 500 1000 2000 4000 6000 10000


# main_reward_coef=1
# LR=1e-5
# TrainingSteps=8000000
# BETA_SMOOTH=20
# # coef_list=(0 5 20 50)

# CUDA_VISIBLE_DEVICES=0 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=1 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=2 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=3 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=4 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=5 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=6 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=7 python main.py --policy_evaluation_new --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 

##### original env & new policy evaluation

# chenghao
# coef: 0 1 2 5 10 20 30 50 100 200 500 1000

# main_reward_coef=1
# LR=1e-5
# TrainingSteps=8000000
# BETA_SMOOTH=20
# # coef_list=(0 1 2 5)

# CUDA_VISIBLE_DEVICES=0 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=1 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=2 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=3 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=4 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=5 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=6 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=7 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original 

main_reward_coef=0
LR=1e-6
TrainingSteps=8000000
BETA_SMOOTH=20
coef_list=(0 5 10 20 50 100 200 500)

CUDA_VISIBLE_DEVICES=0 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=1 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=2 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=3 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=4 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[4]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=5 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[5]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=6 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[6]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=7 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[7]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original 


####### new_PE ends


#### previous

#### original env

# main_reward_coef=0 # 0 for stability test
# LR=1e-6
# TrainingSteps=5000000
# BETA_SMOOTH=20
# coef_list=(5 10 20 50)
# # 5 10 20 50 

# # # one seeds
# CUDA_VISIBLE_DEVICES=4 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=5 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=6 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=7 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original 

# two seeds
# CUDA_VISIBLE_DEVICES=0 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=1 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=2 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=3 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=4 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=5 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=6 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=7 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original 

##### modified env

# main_reward_coef=0 # 0 for stability test
# LR=1e-6
# TrainingSteps=5000000
# BETA_SMOOTH=20
# # coef_list=(10 50 100 200)
# coef_list=(0 1 2 5)

# # one seed

# CUDA_VISIBLE_DEVICES=0 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=1 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=2 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=3 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 



# two seeds

# CUDA_VISIBLE_DEVICES=0 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=1 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=2 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=3 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=4 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=5 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=6 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=7 python main.py --modifedEnv --zeta_0 0 --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 


### modified env
# ours

# LR=1e-5
# BETA_SMOOTH=30
# coef_list=(0 2000 4000 6000)

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
# LR=1e-7

# CUDA_VISIBLE_DEVICES=0 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _lr${LR}_s_2 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _lr${LR}_s_1 &
# CUDA_VISIBLE_DEVICES=7 python main.py --lr $LR --modifedEnv --zeta_0 0 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_2 10 --exp_path_extra _lr${LR}_s_2


### Eric's original env

# ours

# LR=1e-5
# BETA_SMOOTH=30
# coef_list=(0 10 20 50)

# CUDA_VISIBLE_DEVICES=0 python main.py --lr $LR --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr $LR --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr $LR --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr $LR --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --lr $LR --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --lr $LR --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --lr $LR --exp_path_env original \
# --beta_smooth $BETA_SMOOTH --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --lr $LR --exp_path_env original \
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