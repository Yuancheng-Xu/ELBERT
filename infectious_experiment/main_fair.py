import argparse
import copy
import os
# import random
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import torch
# import tqdm
# from absl import flags
# from matplotlib import pyplot as plt
# import matplotlib.patches as mpatches
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

# from networkx.algorithms import community

import sys
sys.path.insert(1, '/cmlscratch/xic/FairRL/')

from infectious_experiment.config_fair import INFECTION_PROBABILITY, INFECTED_EXIT_PROBABILITY, NUM_TREATMENTS, BURNIN, \
    EXP_DIR, POLICY_KWARGS_fair, SAVE_FREQ, GRAPH_NAME, EVAL_INTERVAL
from infectious_experiment.environments import infectious_disease
from infectious_experiment.environments.rewards import InfectiousReward_fair
from infectious_experiment.agents.ppo.ppo_wrapper_env_fair import PPOEnvWrapper_fair
from infectious_experiment.agents.ppo.sb3.ppo_fair import PPO_fair
from infectious_experiment.agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair
from infectious_experiment.agents.ppo.sb3.utils_fair import DummyVecEnv_fair, Monitor_fair

# plot evaluation
from infectious_experiment.plot import plot_return_bias

# Chenghao's env
from infectious_experiment.new_env import create_GeneralInfectiousDiseaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)


GRAPHS = {'karate': nx.karate_club_graph()}



def train(train_timesteps, env, bias_coef, beta_smooth, lr, exp_dir, modifedEnv):

    save_dir = f'{exp_dir}/models/'

    exp_exists = False
    if os.path.isdir(save_dir):
        exp_exists = True
        if input(f'{save_dir} already exists; do you want to retrain / continue training? (y/n): ') != 'y':
            exit()

        print('Training from start...')

    print('env_params: ', env.state.params)


    env = PPOEnvWrapper_fair(env=env, reward_fn=InfectiousReward_fair)
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
                    modifedEnv = modifedEnv)

        shutil.rmtree(exp_dir, ignore_errors=True)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=save_dir,
                                             name_prefix='rl_model')

    model.set_logger(configure(folder=exp_dir))
    model.learn(total_timesteps=train_timesteps, callback=checkpoint_callback, log_interval=50)
    model.save(save_dir + '/final_model')



    
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
    parser.add_argument('--beta_smooth', type=float, default= - 100000) # not used here for num_group = 2

    
    args = parser.parse_args()

    print('\n\n\n',args,'\n\n\n')
    exp_dir  = os.path.join(EXP_DIR,args.exp_path) 
    print('exp_dir:{}'.format(exp_dir))

    if not args.modifedEnv:
        print('Using the original env in Eric\'s code')
        graph = GRAPHS[GRAPH_NAME]
        # Randomly initialize a node to infected
        initial_health_state = [0 for _ in range(graph.number_of_nodes())]
        initial_health_state[0] = 1
        env = infectious_disease.build_sir_model(
            population_graph=graph,
            infection_probability=INFECTION_PROBABILITY,
            infected_exit_probability=INFECTED_EXIT_PROBABILITY,
            num_treatments=NUM_TREATMENTS,
            max_treatments=1,
            burn_in=BURNIN,
            # Treatments turn susceptible people into recovered without having them
            # get sick.
            treatment_transition_matrix=np.array([[0, 0, 1],
                                                  [0, 1, 0],
                                                  [0, 0, 1]]),
            initial_health_state = copy.deepcopy(initial_health_state)
        )
    else:
        print('main_fair.py: Using Chenghao\'s modified env')
        env = create_GeneralInfectiousDiseaseEnv()

    if args.train:
        train(train_timesteps=args.train_timesteps, env=env, bias_coef = args.bias_coef, beta_smooth = args.beta_smooth,\
              lr=args.lr, exp_dir=exp_dir, modifedEnv = args.modifedEnv)
        # plot evaluation
        plot_return_bias(args.exp_path,smooth=2)


if __name__ == '__main__':
    main()