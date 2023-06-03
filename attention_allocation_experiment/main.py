from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
# import random
import shutil
from pathlib import Path
import json

import numpy as np
import torch
# import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

import sys
sys.path.insert(1, '/cmlscratch/xic/FairRL_new/')

from attention_allocation_experiment.config import  EXP_DIR, POLICY_KWARGS_fair, SAVE_FREQ, EVAL_INTERVAL, \
    EP_TIMESTEPS_EVAL, EP_TIMESTEPS, EVAL_NUM_EPS
from environments.attention_allocation import LocationAllocationEnv, Params
from environments.rewards import AttentionAllocationReward

# new multiple reward version: 
from attention_allocation_experiment.agents.ppo.ppo_wrapper_env_fair import PPOEnvWrapper_fair
from attention_allocation_experiment.agents.ppo.sb3.ppo_fair import PPO_fair
from attention_allocation_experiment.agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair
from attention_allocation_experiment.agents.ppo.sb3.utils_fair import DummyVecEnv_fair, Monitor_fair

# plot evaluation
from attention_allocation_experiment.plot import plot_return_bias

# Chenghao's env
from attention_allocation_experiment.new_env import create_GeneralLocationAllocationEnv


device = torch.device("cuda")
print('Using device: ', device)
torch.cuda.empty_cache()

def parser_train():
    parser = argparse.ArgumentParser()

    # our method param
    parser.add_argument('--bias_coef', type=float, default=20) 
    parser.add_argument('--beta_smooth', type=float, default=5) 
    # baseline param
    parser.add_argument('--algorithm', type=str, default='ours', choices=['ours','APPO','GPPO','RPPO']) 
    parser.add_argument('--omega_APPO', type=float, default=0.05) 
    parser.add_argument('--beta_0_APPO', type=float, default=1) 
    parser.add_argument('--beta_1_APPO', type=float, default=0.15) 
    parser.add_argument('--beta_2_APPO', type=float, default=0.15) 
    # training param
    parser.add_argument('--lr', type=float, default=1e-5) 
    parser.add_argument('--train_timesteps', type=int, default=5e6) 
    parser.add_argument('--buffer_size_training', type=int, default=4096)  # only for training; for evaluation, the buffer_size = env.ep_timesteps, the number of steps in one episode
    # base env param
    parser.add_argument('--modifedEnv', action='store_true') # If True, use Chenghao's modifed env; NOTE: will be deprecated
    parser.add_argument('--n_locations', type=int, default=5)
    parser.add_argument('--incident_rates','--list', nargs='+', default=[8, 6, 4, 3, 1.5]) # python main.py -incident_rates 8 6 4 3 1.5
    parser.add_argument('--dynamic_rate', type=float, default=0.1) 
    parser.add_argument('--n_attention_units', type=int, default=6) 
    # env param for wrapper and reward
    parser.add_argument('--include_delta', action='store_false', help='whether include the ratio in the observation space')
    parser.add_argument('--zeta_0', type=float, default=1) 
    parser.add_argument('--zeta_1', type=float, default=0.25) 
    parser.add_argument('--zeta_2', type=float, default=0) # for training (during eval zeta_2 = 0 always). Non-zero for RPPO. 
    # dir name
    parser.add_argument('--exp_path_env', type=str, default='original') # name of env
    parser.add_argument('--exp_path_extra', type=str, default='_s_0/') # including seed

    args = parser.parse_args()
    return args

def organize_param(args):
    '''
    organize the input arguments into groups
    '''
    # check method consistency
    if args.algorithm == 'APPO' or args.algorithm == 'GPPO':
        args.bias_coef = 0 # disable our method
        args.zeta_2 = 0 # disable RPPO
    elif args.algorithm == 'ours':
        assert args.bias_coef > -1e-5, 'bias_coef should be positive when using our method'
        args.zeta_2 = 0 # disable RPPO
    else:
        # RPPO
        assert args.algorithm == 'RPPO', 'Invalid algorithm name. Should be among [ours, APPO, GPPO, RPPO]'
        assert args.zeta_2 >  -1e-5, 'zeta_2 should be positive when using RPPO'
        args.bias_coef = 0 # disable our method

    print('\n\n\n',args,'\n\n\n')
    # our method param
    mitigation_params = {'bias_coef':args.bias_coef, 'beta_smooth':args.beta_smooth}

    # baseline param
    baselines_params = {'method':args.algorithm, 'APPO': args.algorithm == 'APPO', 'OMEGA_APPO': args.omega_APPO, \
                        'BETA_0_APPO':args.beta_0_APPO, 'BETA_1_APPO':args.beta_1_APPO, 'BETA_2_APPO':args.beta_2_APPO}

    # base env param
    env_param_base = {'modifedEnv':args.modifedEnv,
                      'N_LOCATIONS':args.n_locations, 'INCIDENT_RATES':args.incident_rates, 'DYNAMIC_RATE':args.dynamic_rate,\
                      'N_ATTENTION_UNITS':args.n_attention_units}
    # env param for wrapper and reward
    env_param_dict_train = {'include_delta':args.include_delta, 'zeta_0':args.zeta_0, 'zeta_1':args.zeta_1, 'zeta_2':args.zeta_2,\
                      'ep_timesteps':EP_TIMESTEPS}
    env_param_dict_eval = {'include_delta':args.include_delta, 'zeta_0':args.zeta_0, 'zeta_1':args.zeta_1, 'zeta_2':0,\
                      'ep_timesteps':EP_TIMESTEPS_EVAL}
    
    # training param
    training_params = {'lr': args.lr, 'train_timesteps':args.train_timesteps, 'buffer_size_training':args.buffer_size_training}

    # evaluation param
    exp_dir  = get_dir(args)
    eval_kwargs = {'eval_write_path': exp_dir, \
                   'eval_interval':EVAL_INTERVAL, 'num_eps_eval':EVAL_NUM_EPS}
    
    # save args into file
    with open(os.path.join(exp_dir,'params.json'), 'w') as fp:
        for dict_ in [mitigation_params,baselines_params,env_param_base,env_param_dict_train,training_params,eval_kwargs]:
            json.dump(dict_, fp, sort_keys=False, indent=4)

    return mitigation_params, baselines_params, env_param_base, env_param_dict_train, env_param_dict_eval, training_params, eval_kwargs

def get_dir(args):
    '''
    name the experiment directory according to args
    '''
    print('args.exp_path_env :{}'.format(args.exp_path_env))
    exp_dir  = os.path.join(EXP_DIR, args.exp_path_env, args.algorithm)
    if args.algorithm == 'ours':
        exp_dir  = os.path.join(exp_dir, 'Smooth_{}'.format(args.beta_smooth),\
                            'b_{}'.format(args.bias_coef)+args.exp_path_extra)
        print('Using our method with smooth={}, bias_coef={}'.format(args.beta_smooth,args.bias_coef))
    else:
        exp_dir  = os.path.join(exp_dir, args.exp_path_extra)
        print('Using {}'.format(args.algorithm))
    
    if os.path.isdir(exp_dir):
        raise ValueError(f'{exp_dir} already exists; You could delete it manually if you want to train again')
        # print(f'{exp_dir} already exists; You could delete it manually if you want to train again')
    # print('exp_dir:{}'.format(exp_dir))

    shutil.rmtree(exp_dir, ignore_errors=True) # clear the file first
    save_dir = f'{exp_dir}/models/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def train(env, mitigation_params, baselines_params, env_param_dict_train, env_param_dict_eval, training_params, eval_kwargs):

    env_train = PPOEnvWrapper_fair(env=copy.deepcopy(env), reward_fn=AttentionAllocationReward, env_param_dict = env_param_dict_train)
    env_train = Monitor_fair(env_train)
    env_train = DummyVecEnv_fair([lambda: env_train]) 

    env_eval = PPOEnvWrapper_fair(env=copy.deepcopy(env), reward_fn=AttentionAllocationReward, env_param_dict = env_param_dict_eval)
    eval_kwargs['env_eval'] = env_eval
   
    model = PPO_fair(ActorCriticPolicy_fair, env_train,
                policy_kwargs=POLICY_KWARGS_fair,
                verbose=1,
                learning_rate = training_params['lr'],
                n_steps = training_params['buffer_size_training'], 
                device=device,

                mitigation_params = mitigation_params,
                baselines_params = baselines_params, 
                eval_kwargs = eval_kwargs,
                )

    exp_dir = eval_kwargs['eval_write_path']
    save_dir = f'{exp_dir}/models/'
    # shutil.rmtree(exp_dir, ignore_errors=True) # clear the file first
    # Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=save_dir,
                                             name_prefix='rl_model')
    model.set_logger(configure(folder=exp_dir))

    model.learn(total_timesteps=training_params['train_timesteps'], callback=checkpoint_callback) # actual training
    model.save(save_dir + '/final_model')


def main():
    args = parser_train()

    mitigation_params, baselines_params, env_param_base, env_param_dict_train, env_param_dict_eval, training_params, eval_kwargs = \
    organize_param(args)
    
    if not args.modifedEnv:
        print('Using the original env in Eric\'s code')
        env_params = Params(
            n_locations=env_param_base['N_LOCATIONS'],
            prior_incident_counts=tuple(500 for _ in range(env_param_base['N_LOCATIONS'])),
            incident_rates=env_param_base['INCIDENT_RATES'],
            n_attention_units=env_param_base['N_ATTENTION_UNITS'],
            miss_incident_prob=tuple(0. for _ in range(env_param_base['N_LOCATIONS'])),
            extra_incident_prob=tuple(0. for _ in range(env_param_base['N_LOCATIONS'])),
            dynamic_rate=env_param_base['DYNAMIC_RATE'])
        env = LocationAllocationEnv(params=env_params)
    else:
        print('main.py: Using Chenghao\'s modified env')
        env = create_GeneralLocationAllocationEnv()

    train(env = env, mitigation_params = mitigation_params, baselines_params = baselines_params, env_param_dict_train = env_param_dict_train, \
          env_param_dict_eval = env_param_dict_eval, training_params = training_params, eval_kwargs = eval_kwargs)

    # plot evaluation
    plot_return_bias(eval_kwargs['eval_write_path'], smooth=2)
        

if __name__ == '__main__':
    main()