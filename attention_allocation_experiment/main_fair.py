from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
# import random
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

from attention_allocation_experiment.config_fair import N_LOCATIONS, INCIDENT_RATES, N_ATTENTION_UNITS, DYNAMIC_RATE, \
    EXP_DIR, POLICY_KWARGS_fair, SAVE_FREQ, EVAL_INTERVAL \
    # TRAIN_TIMESTEPS, LEARNING_RATE
from environments.attention_allocation import LocationAllocationEnv, Params
from environments.rewards import AttentionAllocationReward_fair

from graphing.plot_att_all_over_time_across_agents import plot_att_all_over_time_across_agents
from graphing.plot_deltas_over_time_across_agents import plot_deltas_over_time_across_agents
from graphing.plot_incidents_missed_over_time_across_agents import plot_incidents_missed_over_time_across_agents
from graphing.plot_incidents_seen_over_time_across_agents import plot_incidents_seen_over_time_across_agents
# from graphing.plot_rews import plot_rews
from graphing.plot_rew_over_time_across_agents import plot_rew_over_time_across_agents
from graphing.plot_rew_terms_over_time_across_agents import plot_rew_terms_over_time_across_agents
from graphing.plot_true_rates_over_time_across_agents import plot_true_rates_over_time_across_agents

# new multiple reward version: 
from attention_allocation_experiment.agents.ppo.ppo_wrapper_env_fair import PPOEnvWrapper_fair
from attention_allocation_experiment.agents.ppo.sb3.ppo_fair import PPO_fair
from attention_allocation_experiment.agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair
from attention_allocation_experiment.agents.ppo.sb3.utils_fair import DummyVecEnv_fair, Monitor_fair

# plot evaluation
from attention_allocation_experiment.plot import plot_return_bias

# Chenghao's env
from attention_allocation_experiment.new_env import create_GeneralLocationAllocationEnv



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print('Using device: ', device)
torch.cuda.empty_cache()



def train(train_timesteps, env, bias_coef, beta_smooth, lr, exp_dir, modifedEnv=False, seed = None):

    save_dir = f'{exp_dir}/models/'

    exp_exists = False
    if os.path.isdir(save_dir):
        exp_exists = True
        if input(f'{save_dir} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
            exit()

        print('Training from start...')

    print('env_params: ', env.state.params)


    env = PPOEnvWrapper_fair(env=env, reward_fn=AttentionAllocationReward_fair)
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
                    beta_smooth = beta_smooth,
                    bias_coef=bias_coef,
                    eval_write_path = os.path.join(exp_dir,'eval.csv'),
                    eval_interval = EVAL_INTERVAL,
                    modifedEnv = modifedEnv,
                    seed = seed)

        shutil.rmtree(exp_dir, ignore_errors=True)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=save_dir,
                                             name_prefix='rl_model')

    model.set_logger(configure(folder=exp_dir))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback)
    model.save(save_dir + '/final_model')


# TODO: Have not modified this function yet. Maybe don't need to 
def display_eval_results(eval_dir):
    tot_eval_data = {}
    agent_names = copy.deepcopy(next(os.walk(eval_dir))[1])
    for agent_name in agent_names:
        tot_eval_data[agent_name] = {}
        for key in next(os.walk(f'{eval_dir}/{agent_name}'))[2]:
            key = key.split('.npy')[0]
            tot_eval_data[agent_name][key] = np.load(f'{eval_dir}/{agent_name}/{key}.npy', allow_pickle=True)

    # Plot all human_designed_policies evaluations
    plot_rew_over_time_across_agents(tot_eval_data)
    plot_incidents_seen_over_time_across_agents(tot_eval_data)
    plot_incidents_missed_over_time_across_agents(tot_eval_data)
    plot_att_all_over_time_across_agents(tot_eval_data)
    plot_true_rates_over_time_across_agents(tot_eval_data)
    plot_deltas_over_time_across_agents(tot_eval_data)
    plot_rew_terms_over_time_across_agents(tot_eval_data)




def main():
    parser = argparse.ArgumentParser()
    # essential
    parser.add_argument('--bias_coef', type=float, default=0.0)
    parser.add_argument('--beta_smooth', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-6) # Eric: 1e-5
    parser.add_argument('--train_timesteps', type=int, default=5e6) 
    parser.add_argument('--seed', type=int, default=123)
    # evaluation
    parser.add_argument('--exp_path', type=str, default='debug/') # experiment result path exp_dir will be EXP_DIR/exp_path

    parser.add_argument('--modifedEnv', action='store_true') # If True, use Chenghao's modifed env

    # others
    parser.add_argument('--train', action='store_false')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo']) # later, change this to ppo_fair

    
    args = parser.parse_args()

    print('\n\n\n',args,'\n\n\n')
    exp_dir  = os.path.join(EXP_DIR,args.exp_path) 
    print('exp_dir:{}'.format(exp_dir))

    if not args.modifedEnv:
        print('Using the original env in Eric\'s code')
        env_params = Params(
            n_locations=N_LOCATIONS,
            prior_incident_counts=tuple(500 for _ in range(N_LOCATIONS)),
            incident_rates=INCIDENT_RATES,
            n_attention_units=N_ATTENTION_UNITS,
            miss_incident_prob=tuple(0. for _ in range(N_LOCATIONS)),
            extra_incident_prob=tuple(0. for _ in range(N_LOCATIONS)),
            dynamic_rate=DYNAMIC_RATE)
        env = LocationAllocationEnv(params=env_params)
    else:
        print('main_fair.py: Using Chenghao\'s modified env')
        env = create_GeneralLocationAllocationEnv()

    if args.train:
        
        train(train_timesteps=args.train_timesteps, env=env, bias_coef = args.bias_coef, beta_smooth = args.beta_smooth,\
              lr=args.lr, exp_dir=exp_dir, modifedEnv = args.modifedEnv, seed = args.seed)

        # plot evaluation
        plot_return_bias(args.exp_path,smooth=2)


        

if __name__ == '__main__':
    main()