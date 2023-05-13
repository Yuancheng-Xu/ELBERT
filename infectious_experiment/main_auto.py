import argparse
import copy
import os
import random
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import tqdm
from absl import flags
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append('..')

from infectious_experiment.config_auto import INFECTION_PROBABILITY, NUM_TREATMENTS, BURNIN, \
    EXP_DIR_AUTO, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ_AUTO, EVAL_FREQ_AUTO, TRAIN_TIMESTEPS, GRAPH_NAME, \
    INFECTED_EXIT_PROBABILITY
from infectious_experiment.environments.infectious_disease import InfectiousDiseaseEnv, Params
from infectious_experiment.environments.rewards import InfectiousReward, calc_percent_healthy
from infectious_experiment.agents.ppo.ppo_wrapper_env_auto import PPOEnvWrapper_auto
from infectious_experiment.agents.ppo.sb3.ppo_auto import PPO_auto

# Chenghao's env
from infectious_experiment.new_env import create_GeneralInfectiousDiseaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)


GRAPHS = {'karate': nx.karate_club_graph()}


def train_eval_auto(train_timesteps, env, env_params, lr, name, seed,
                    eval_interval=EVAL_FREQ_AUTO, 
                    eval_eps=100, 
                    eval_timesteps=20,
                    eval_env_name="mod"):
    ori_dir = "%s_%s_%s_%d" % (EXP_DIR_AUTO[name], eval_env_name, lr, seed)
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
                        reward_fn=InfectiousReward,
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
                eval_timesteps=eval_timesteps,
                eval_env_name=eval_env_name
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

    graph = GRAPHS[GRAPH_NAME]
    # Randomly initialize a node to infected
    initial_health_state = [0 for _ in range(graph.number_of_nodes())]
    initial_health_state[0] = 1
    transition_matrix = np.array([
        [0, 0, 0],
        [0, 1 - INFECTED_EXIT_PROBABILITY, INFECTED_EXIT_PROBABILITY],
        [0, 0, 1]])

    env_params = Params(
        population_graph=graph,
        transition_matrix=transition_matrix,
        treatment_transition_matrix=np.array([[0, 0, 1],
                                              [0, 1, 0],
                                              [0, 0, 1]]),
        state_names=['susceptible', 'infected', 'recovered'],
        healthy_index=0,
        infectious_index=1,
        healthy_exit_index=1,
        infection_probability=INFECTION_PROBABILITY,
        initial_health_state=copy.deepcopy(initial_health_state),
        initial_health_state_seed=100,
        num_treatments=NUM_TREATMENTS,
        max_treatments=1,
        burn_in=BURNIN)

    if not args.modifedEnv:
        print('Using the original env in Eric\'s code')
        env_name = 'ori'
        env = InfectiousDiseaseEnv(env_params)
    else:
        print('main_fair.py: Using Chenghao\'s modified env')
        env_name = 'mod'
        env = create_GeneralInfectiousDiseaseEnv()

    if args.train:
        train_eval_auto(train_timesteps=args.train_steps, 
                        env=env,
                        env_params=env_params, 
                        lr=args.lr, 
                        name=args.algorithm,
                        seed=args.seed,
                        eval_interval=EVAL_FREQ_AUTO,
                        eval_eps=100,
                        eval_timesteps=20,
                        eval_env_name=env_name,
                    )

if __name__ == '__main__':
    main()