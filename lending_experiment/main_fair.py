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

# from yaml import full_load

import sys
sys.path.insert(1, '/cmlscratch/xic/FairRL/')

from lending_experiment.config_fair import CLUSTER_PROBABILITIES, GROUP_0_PROB, BANK_STARTING_CASH, INTEREST_RATE, \
    CLUSTER_SHIFT_INCREMENT, EP_TIMESTEPS, NUM_GROUPS, TRAIN_TIMESTEPS, SAVE_DIR, EXP_DIR, POLICY_KWARGS_fair, \
        LEARNING_RATE, SAVE_FREQ, EVAL_MODEL_PATHS
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
torch.cuda.empty_cache()



def train(train_timesteps, env, bias_coef):

    exp_exists = False
    if os.path.isdir(SAVE_DIR):
        exp_exists = True
        if input(f'{SAVE_DIR} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
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
        model_name = input(f'Specify the model you would like to load in. Do not include the .zip: ')
        model = PPO_fair.load(EXP_DIR + "models/" + model_name, verbose=1, device=device)
        model.set_env(env)
    else:        
        print('PPO_fair: bias_coef:{}'.format(bias_coef))
        model = PPO_fair(ActorCriticPolicy_fair, env,
                    policy_kwargs=POLICY_KWARGS_fair,
                    verbose=1,
                    learning_rate=LEARNING_RATE,
                    device=device,
                    bias_coef=bias_coef)

        shutil.rmtree(EXP_DIR, ignore_errors=True)
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR,
                                             name_prefix='rl_model')

    model.set_logger(configure(folder=EXP_DIR))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
    model.save(SAVE_DIR + '/final_model')

    # Once we finish learning, plot the returns over time and save into the experiments directory
    plot_rets(EXP_DIR)



# xyc: not finished
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
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'cpo']) # later, change this to ppo_fair
    parser.add_argument('--eval_path', dest='eval_path', type=str, default=None)
    parser.add_argument('--display_eval_path', dest='display_eval_path', type=str, default=None)
    parser.add_argument('--show_train_progress', action='store_true')

    parser.add_argument('--bias_coef', type=float, default=0.5)
    args = parser.parse_args()

    print('\n\n\n',args,'\n\n\n')

    env_params = DelayedImpactParams(
        applicant_distribution=two_group_credit_clusters(
            cluster_probabilities=CLUSTER_PROBABILITIES,
            group_likelihoods=[GROUP_0_PROB, 1 - GROUP_0_PROB]),
        bank_starting_cash=BANK_STARTING_CASH,
        interest_rate=INTEREST_RATE,
        cluster_shift_increment=CLUSTER_SHIFT_INCREMENT,
    )
    env = DelayedImpactEnv(env_params)

    if args.train:
        train(train_timesteps=TRAIN_TIMESTEPS, env=env, bias_coef = args.bias_coef)
        plot_rets(exp_path=EXP_DIR, save_png=True)

    if args.show_train_progress:
        plot_rets(exp_path=EXP_DIR, save_png=False)

    if args.display_eval_path is not None:
        display_eval_results(eval_dir=args.display_eval_path)

    if args.eval_path is not None:

        assert(args.eval_path is not None)
        p = Path(args.eval_path)
        if p.exists():
            resp = input(f'{args.eval_path} already exists; do you want to override it? (y/n): ')
            if resp != 'y':
                exit('Exiting.')

        # Initialize eval directory to store eval information
        shutil.rmtree(args.eval_path, ignore_errors=True)
        Path(args.eval_path).mkdir(parents=True, exist_ok=True)

        # Get random seeds
        eval_eps = 10
        eval_timesteps = 10000
        raise ValueError('Why eval_timesteps = 10000???')
        seeds = [random.randint(0, 10000) for _ in range(eval_eps)]

        with open(args.eval_path + '/seeds.txt', 'w') as f:
            f.write(str(seeds))

        # First, evaluate PPO human_designed_policies
        for name, model_path in EVAL_MODEL_PATHS.items():
            env = DelayedImpactEnv(env_params)
            agent = PPO.load(model_path, verbose=1)
            evaluate(env=PPOEnvWrapper_fair(env=env, reward_fn=LendingReward, ep_timesteps=eval_timesteps), # here number of steps per episode is overidden as eval_timesteps.
                     agent=agent,
                     num_eps=eval_eps,
                     num_timesteps=eval_timesteps,
                     name=name,
                     seeds=seeds,
                     eval_path=args.eval_path)

        

        


if __name__ == '__main__':
    main()