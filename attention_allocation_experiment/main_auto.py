import argparse
import copy
import random
import time
import os
import shutil
from pathlib import Path

import numpy as np
# from sympy import E
import torch
import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append('..')

from attention_allocation_experiment.agents.human_designed_policies.allocation_agents import MLEGreedyAgent, MLEGreedyAgentParams
from config_auto import N_LOCATIONS, INCIDENT_RATES, N_ATTENTION_UNITS, DYNAMIC_RATE, \
    EXP_DIR_AUTO, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ_AUTO, TRAIN_TIMESTEPS, EVAL_FREQ_AUTO
from new_env import GeneralParams, create_GeneralLocationAllocationEnv
from environments.attention_allocation import LocationAllocationEnv, Params
from environments.rewards import AttentionAllocationReward
from attention_allocation_experiment.agents.ppo.ppo_wrapper_env_auto import PPOEnvWrapper_auto
from attention_allocation_experiment.agents.ppo.sb3.ppo_auto import PPO_auto


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)


def train_eval_auto(train_timesteps, env, env_params, lr, name, seed,
                    eval_interval=EVAL_FREQ_AUTO, 
                    eval_eps=3, 
                    eval_timesteps=1000,
                    eval_env="mod"):
    ori_dir = "%s_%s_%s_%d" % (EXP_DIR_AUTO[name], lr, eval_env, seed)
    save_dir = '%s/models/' % (ori_dir)
    eval_dir = '%s/eval/' % (ori_dir)

    exp_exists = False
    if os.path.isdir(save_dir):
        exp_exists = True
        # if input(f'{save_dir} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
        #     exit()

        print('Training from start...')

    print('env_params: ', env.state.params)

    env = PPOEnvWrapper_auto(env=env, 
                        reward_fn=AttentionAllocationReward,
                        name=name
                    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = None
    should_load = False
    # if exp_exists:
    #     resp = input(f'\nWould you like to load the previous model to continue training? If you do not select yes, you will start a new training. (y/n): ')
    #     if resp != 'y' and resp != 'n':
    #         exit('Invalid response for resp: ' + resp)
    #     should_load = resp == 'y'


    model = PPO_auto("MlpPolicy", env,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                learning_rate=lr,
                device=device,
                name=name,
                eval_env_params=env_params,
                eval_write_path=eval_dir,
                eval_interval=eval_interval,
                eval_eps=eval_eps,
                eval_timesteps=eval_timesteps 
                )

    shutil.rmtree(ori_dir, ignore_errors=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ_AUTO, save_path=save_dir,
                                             name_prefix='rl_model')

    model.set_logger(configure(folder=ori_dir))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
    model.save(save_dir + '/final_model')

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--train_steps', type=float, default=TRAIN_TIMESTEPS)
    parser.add_argument('--algorithm', type=str, default='A-PPO', choices=['A-PPO', 'R-PPO', 'G-PPO'])
    
    parser.add_argument('--modifedEnv', action='store_true') # If True, use Chenghao's modifed env

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    env_params = Params(
        n_locations=N_LOCATIONS,
        prior_incident_counts=tuple(500 for _ in range(N_LOCATIONS)),
        incident_rates=INCIDENT_RATES,
        n_attention_units=N_ATTENTION_UNITS,
        miss_incident_prob=tuple(0. for _ in range(N_LOCATIONS)),
        extra_incident_prob=tuple(0. for _ in range(N_LOCATIONS)),
        dynamic_rate=DYNAMIC_RATE)

    # Initialize the environment

    if args.modifedEnv:
        print('main_fair.py: Using Chenghao\'s modified env')
        env = create_GeneralLocationAllocationEnv()
        env_name= "mod"
    else:
        print('Using the original env in Eric\'s code')
        env = LocationAllocationEnv(params=env_params)
        env_name= "ori"

    if args.train:
        train_eval_auto(train_timesteps=args.train_steps, 
                        env=env,
                        env_params=env_params, 
                        lr=args.lr, 
                        name=args.algorithm,
                        seed=args.seed,
                        eval_interval=EVAL_FREQ_AUTO,
                        eval_eps=3,
                        eval_timesteps=1000,
                        eval_env=env_name
                    )

if __name__ == '__main__':
    main()