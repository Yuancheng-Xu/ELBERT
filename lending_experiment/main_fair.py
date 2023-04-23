from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
# import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.monitor import Monitor # does not deal with multiple rewards
# from stable_baselines3.common.vec_env import DummyVecEnv # does not deal with multiple rewards


import sys
sys.path.insert(1, '/cmlscratch/xic/FairRL/')

from lending_experiment.config_fair import CLUSTER_PROBABILITIES, GROUP_0_PROB, BANK_STARTING_CASH, INTEREST_RATE, \
    CLUSTER_SHIFT_INCREMENT, EXP_DIR, POLICY_KWARGS_fair, \
    SAVE_FREQ, EVAL_INTERVAL \
    # TRAIN_TIMESTEPS, LEARNING_RATE

from lending_experiment.environments.lending import DelayedImpactEnv

from lending_experiment.environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from lending_experiment.environments.rewards import LendingReward_fair

from lending_experiment.graphing.plot_bank_cash_over_time import plot_bank_cash_over_time
# from lending_experiment.graphing.plot_confusion_matrix_over_time import plot_confusion_matrix_over_time
# from lending_experiment.graphing.plot_fn_over_time import plot_fn_over_time
# from lending_experiment.graphing.plot_fp_over_time import plot_fp_over_time
from lending_experiment.graphing.plot_loans_over_time import plot_loans_over_time
from lending_experiment.graphing.plot_rets import plot_rets
from lending_experiment.graphing.plot_rews_over_time import plot_rews_over_time
# from lending_experiment.graphing.plot_tn_over_time import plot_tn_over_time
# from lending_experiment.graphing.plot_tp_over_time import plot_tp_over_time
from lending_experiment.graphing.plot_tpr_gap_over_time import plot_tpr_gap_over_time
from lending_experiment.graphing.plot_tpr_over_time import plot_tpr_over_time

# new multiple reward version: 
from lending_experiment.agents.ppo.ppo_wrapper_env_fair import PPOEnvWrapper_fair
from lending_experiment.agents.ppo.sb3.ppo_fair import PPO_fair
from lending_experiment.agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair
from lending_experiment.agents.ppo.sb3.utils_fair import DummyVecEnv_fair, Monitor_fair
# plot evaluation
from lending_experiment.plot import plot_cash_bias

# Chenghao's env
from lending_experiment.new_env import create_GeneralDelayedImpactEnv



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()



def train(train_timesteps, env, bias_coef, lr, exp_dir):

    save_dir = f'{exp_dir}/models/'

    exp_exists = False
    if os.path.isdir(save_dir):
        exp_exists = True
        if input(f'{save_dir} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
            exit()

        print('Training from start...')

    print('env_params: ', env.state.params)


    env = PPOEnvWrapper_fair(env=env, reward_fn=LendingReward_fair)
    env = Monitor_fair(env)
    env = DummyVecEnv_fair([lambda: env]) 

    
    model = None
    should_load = False
    if exp_exists:
        resp = input(f'\nWould you like to load the previous model to continue training? If you do not select yes, you will start a new training. (y/n): ')
        if resp != 'y' and resp != 'n':
            exit('Invalid response for resp: ' + resp)
        should_load = resp == 'y'

    if should_load:
        raise ValueError('Not implemented yet')
        model_name = input(f'Specify the model you would like to load in. Do not include the .zip: ')
        model = PPO_fair.load(exp_dir + "models/" + model_name, verbose=1, device=device)
        model.set_env(env)
    else:        
        print('PPO_fair: bias_coef:{}'.format(bias_coef))
        model = PPO_fair(ActorCriticPolicy_fair, env,
                    policy_kwargs=POLICY_KWARGS_fair,
                    verbose=1,
                    learning_rate = lr,
                    device=device,
                    bias_coef=bias_coef,
                    eval_write_path = os.path.join(exp_dir,'eval.csv'),
                    eval_interval = EVAL_INTERVAL)

        shutil.rmtree(exp_dir, ignore_errors=True)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=save_dir,
                                             name_prefix='rl_model')

    model.set_logger(configure(folder=exp_dir))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
    model.save(save_dir + '/final_model')

    # Once we finish learning, plot the returns over time and save into the experiments directory
    plot_rets(exp_dir)



# TODO: Have not modified this function yet. Maybe don't need to 
def display_eval_results(eval_dir):
    tot_eval_data = {}
    agent_names = copy.deepcopy(next(os.walk(eval_dir))[1])
    for agent_name in agent_names:
        tot_eval_data[agent_name] = {}
        for key in next(os.walk(f'{eval_dir}/{agent_name}'))[2]:
            key = key.split('.npy')[0]
            tot_eval_data[agent_name][key] = np.load(f'{eval_dir}/{agent_name}/{key}.npy', allow_pickle=True)


    # Plot all agent evaluations
    plot_rews_over_time(tot_eval_data)
    plot_loans_over_time(tot_eval_data)
    plot_bank_cash_over_time(tot_eval_data)
    plot_tpr_over_time(tot_eval_data)
    plot_tpr_gap_over_time(tot_eval_data)




def main():
    parser = argparse.ArgumentParser()
    # essential
    parser.add_argument('--bias_coef', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-6) # Eric: 1e-5
    parser.add_argument('--train_timesteps', type=int, default=5e6) 

    parser.add_argument('--modifedEnv', action='store_true') # If True, use Chenghao's modifed env

    # evaluation
    parser.add_argument('--exp_path', type=str, default='lr_1e-6/bias_1') # experiment result path exp_dir will be EXP_DIR/exp_path

    # others
    parser.add_argument('--train', action='store_false')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo']) # later, change this to ppo_fair
    parser.add_argument('--display_eval_path', dest='display_eval_path', type=str, default=None)
    parser.add_argument('--show_train_progress', action='store_false')

    
    args = parser.parse_args()

    print('\n\n\n',args,'\n\n\n')
    exp_dir  = os.path.join(EXP_DIR,args.exp_path) 
    print('exp_dir:{}'.format(exp_dir))

    if not args.modifedEnv:
        print('Using the original env in Eric\'s code')
        env_params = DelayedImpactParams(
            applicant_distribution=two_group_credit_clusters(
                cluster_probabilities=CLUSTER_PROBABILITIES,
                group_likelihoods=[GROUP_0_PROB, 1 - GROUP_0_PROB]),
            bank_starting_cash=BANK_STARTING_CASH,
            interest_rate=INTEREST_RATE,
            cluster_shift_increment=CLUSTER_SHIFT_INCREMENT,
        )

        env = DelayedImpactEnv(env_params)
    else:
        print('main_fair.py: Using Chenghao\'s modified env')
        env = create_GeneralDelayedImpactEnv()


    if args.train:
        
        train(train_timesteps=args.train_timesteps, env=env, bias_coef = args.bias_coef, lr=args.lr, exp_dir=exp_dir)

        # plot evaluation
        plot_cash_bias(args.exp_path,smooth=2)

        plot_rets(exp_path=exp_dir, save_png=True)

    if args.show_train_progress:
        plot_rets(exp_path=exp_dir, save_png=False)

    if args.display_eval_path is not None:
        display_eval_results(eval_dir=args.display_eval_path)

        

if __name__ == '__main__':
    main()