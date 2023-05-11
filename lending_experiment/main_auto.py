from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import functools
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from absl import flags
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append('..')

from lending_experiment.agents.human_designed_policies import oracle_lending_agent
from lending_experiment.agents.human_designed_policies.classifier_agents import ScoringAgentParams
from lending_experiment.agents.human_designed_policies.threshold_policies import ThresholdPolicy
from lending_experiment.config_auto import CLUSTER_PROBABILITIES, GROUP_0_PROB, BANK_STARTING_CASH, INTEREST_RATE, \
    CLUSTER_SHIFT_INCREMENT, BURNIN, MAXIMIZE_REWARD, EQUALIZE_OPPORTUNITY, EP_TIMESTEPS, NUM_GROUPS, EVAL_ZETA_0, \
    EVAL_ZETA_1, TRAIN_TIMESTEPS, EXP_DIR_AUTO, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ_AUTO, EVAL_FREQ_AUTO, DELAYED_IMPACT_CLUSTER_PROBS
from lending_experiment.new_env import GeneralDelayedImpactEnv, create_GeneralDelayedImpactEnv
from lending_experiment.environments.lending import DelayedImpactEnv
from lending_experiment.environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from lending_experiment.environments.rewards import LendingReward
from lending_experiment.graphing.plot_rets import plot_rets
from lending_experiment.agents.ppo.ppo_wrapper_env_auto import PPOEnvWrapper_auto
from lending_experiment.agents.ppo.sb3.ppo_auto import PPO_auto

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()

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
        # if input(f'{SAVE_DIR} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
        #     exit()

        print('Training from start...')

    print('env_params: ', env.state.params)

    env = PPOEnvWrapper_auto(env=env, reward_fn=LendingReward, name=name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

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

    env_params = DelayedImpactParams(
            applicant_distribution=two_group_credit_clusters(
                cluster_probabilities=DELAYED_IMPACT_CLUSTER_PROBS[0],
                group_likelihoods=[GROUP_0_PROB, 1 - GROUP_0_PROB]),
            bank_starting_cash=BANK_STARTING_CASH,
            interest_rate=INTEREST_RATE,
            cluster_shift_increment=CLUSTER_SHIFT_INCREMENT,
        )
    if args.modifedEnv:
        print('main_fair.py: Using Chenghao\'s modified env')
        env = create_GeneralDelayedImpactEnv()
        env_name= "mod"
    else:
        print('Using the original env in Eric\'s code')
        env = DelayedImpactEnv(env_params)
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
                        eval_timesteps=10000,
                        eval_env=env_name
                    )


if __name__ == '__main__':
    main()