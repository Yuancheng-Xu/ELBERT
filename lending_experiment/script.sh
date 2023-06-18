#!/usr/bin/env bash

trap 'kill 0' SIGINT

# GPPO and APPO: specifying --algorithm GPPO/APPO is enough
# RPPO: also need to specify --zeta_1
# ours: specify --bias_coef (no need for beta_smooth since only 2 groups)

##### modified env & new policy evaluation

# main_reward_coef=1
# LR=1e-6
# TrainingSteps=10000000
# coef_list=(0 10 200 1000) # 0 10 200 1000 2000 5000 10000 20000

# CUDA_VISIBLE_DEVICES=0 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=1 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=2 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=3 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=4 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=5 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=6 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=7 python main.py --policy_evaluation_new --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 


##### original env & new policy evaluation

main_reward_coef=1
LR=1e-5
TrainingSteps=10000000
coef_list=(30000 50000 100000 200000) # 0 10 200 1000 2000 5000 10000 20000 & 30000 50000 100000 200000

CUDA_VISIBLE_DEVICES=0 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=1 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
CUDA_VISIBLE_DEVICES=2 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=3 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
CUDA_VISIBLE_DEVICES=4 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=5 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
CUDA_VISIBLE_DEVICES=6 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
CUDA_VISIBLE_DEVICES=7 python main.py --policy_evaluation_new --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
--bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original 











####### bias stability test begins #######
# NOTE: can decrease EVAL_INTERVAL in config.py if TrainingSteps is very small

# original env
# main_reward_coef=0
# LR=1e-6
# TrainingSteps=5000000
# coef_list=(0 200 1000 2000)


# CUDA_VISIBLE_DEVICES=0 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=1 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=2 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=3 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=4 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=5 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=6 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env original &
# CUDA_VISIBLE_DEVICES=7 python main.py --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env original 



# modified env
# main_reward_coef=0
# LR=1e-6
# TrainingSteps=5000000
# coef_list=(0 200 1000 2000)


# CUDA_VISIBLE_DEVICES=0 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=1 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[0]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=2 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=3 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[1]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=4 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=5 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[2]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=6 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_0 --exp_path_env Chenghao_env_05_14 &
# CUDA_VISIBLE_DEVICES=7 python main.py --modifedEnv --main_reward_coef $main_reward_coef --lr $LR --train_timesteps $TrainingSteps \
# --bias_coef ${coef_list[3]} --exp_path_extra _lr${LR}_s_1 --exp_path_env Chenghao_env_05_14 


####### bias stability test ends #######










### modified env: 1e-6

# ours
# CUDA_VISIBLE_DEVICES=0 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 0 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 0 --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 200 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 200 --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 1000 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 1000 --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 2000 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 2000 --exp_path_extra _lr1e-6_s_1 

# baselines
# CUDA_VISIBLE_DEVICES=0 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _lr1e-6_s_2 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _lr1e-6_s_1 &
# CUDA_VISIBLE_DEVICES=7 python main.py --lr 1e-6 --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _lr1e-6_s_2

### modified env (default: 1e-5)

# ours
# CUDA_VISIBLE_DEVICES=0 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 0 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 0 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 200 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 200 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 1000 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 1000 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 2000 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --bias_coef 2000 --exp_path_extra _s_1 

# baselines
# CUDA_VISIBLE_DEVICES=0 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm GPPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm APPO --exp_path_extra _s_2 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _s_1 &
# CUDA_VISIBLE_DEVICES=7 python main.py --modifedEnv --zeta_0 1 --exp_path_env Chenghao_env_05_14 \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _s_2

### Eric's original env (lr: 1e-6)

# ours
# CUDA_VISIBLE_DEVICES=0 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 0 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 0 --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 200 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 200 --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=4 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 1000 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=5 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 1000 --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=6 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 2000 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=7 python main.py --lr 1e-6 --exp_path_env original \
# --bias_coef 2000 --exp_path_extra _lr1e-6_s_1 

# baselines
# CUDA_VISIBLE_DEVICES=0 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm GPPO --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm GPPO --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm APPO --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm APPO --exp_path_extra _lr1e-6_s_1 & 
# CUDA_VISIBLE_DEVICES=0 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm APPO --exp_path_extra _lr1e-6_s_2 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _lr1e-6_s_0 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _lr1e-6_s_1 &
# CUDA_VISIBLE_DEVICES=3 python main.py --lr 1e-6 --exp_path_env original \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _lr1e-6_s_2

### Eric's original env (default: 1e-5)

# ours
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --bias_coef 0 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --bias_coef 0 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --bias_coef 200 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --bias_coef 200 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --bias_coef 1000 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --bias_coef 1000 --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --bias_coef 2000 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --bias_coef 2000 --exp_path_extra _s_1 

# baselines
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --algorithm GPPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --algorithm GPPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --algorithm APPO --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --algorithm APPO --exp_path_extra _s_1 & 
# CUDA_VISIBLE_DEVICES=0 python main.py --exp_path_env original \
# --algorithm APPO --exp_path_extra _s_2 & 
# CUDA_VISIBLE_DEVICES=1 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _s_0 & 
# CUDA_VISIBLE_DEVICES=2 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _s_1 &
# CUDA_VISIBLE_DEVICES=3 python main.py --exp_path_env original \
# --algorithm RPPO --zeta_1 2 --exp_path_extra _s_2