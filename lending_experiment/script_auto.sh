source ~/.bashrc
conda activate pocar
cd /cmlscratch/dengch/fairness_RL/Bias-Mitigation-RL/lending_experiment

expname0=train_gppo
expname1=train_appo
expname2=train_rppo
expname3=train_gppo_mod
expname4=train_appo_mod
expname5=train_rppo_mod

trap 'kill 0' SIGINT

export CUDA_VISIBLE_DEVICES=0

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm G-PPO \
                            --modifedEnv \
                            > auto_logs/${expname0}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm A-PPO \
                            --modifedEnv \
                            > auto_logs/${expname1}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm R-PPO \
                            --modifedEnv \
                            > auto_logs/${expname2}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm G-PPO \
                            > auto_logs/${expname3}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1

python main_auto.py     --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm A-PPO \
                            > auto_logs/${expname4}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm R-PPO \
                            > auto_logs/${expname5}.log 2>&1