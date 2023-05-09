source ~/.bashrc
conda activate pocar
cd /cmlscratch/dengch/fairness_RL/Bias-Mitigation-RL/attention_allocation_experiment

seed=0

expname0=train_gppo_${seed}
expname1=train_appo_${seed}
expname2=train_rppo_${seed}
expname3=train_gppo_mod_${seed}
expname4=train_appo_mod_${seed}
expname5=train_rppo_mod_${seed}

trap 'kill 0' SIGINT

export CUDA_VISIBLE_DEVICES=0

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm G-PPO \
                            --modifedEnv \
                            --seed ${seed} \
                            > auto_logs/${expname0}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm A-PPO \
                            --modifedEnv \
                            --seed ${seed} \
                            > auto_logs/${expname1}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm R-PPO \
                            --modifedEnv \
                            --seed ${seed} \
                            > auto_logs/${expname2}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm G-PPO \
                            --seed ${seed} \
                            > auto_logs/${expname3}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm A-PPO \
                            --seed ${seed} \
                            > auto_logs/${expname4}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5

python main_auto.py         --train \
                            --lr 1e-6 \
                            --train_steps 5e6 \
                            --algorithm R-PPO \
                            --seed ${seed} \
                            > auto_logs/${expname5}.log 2>&1