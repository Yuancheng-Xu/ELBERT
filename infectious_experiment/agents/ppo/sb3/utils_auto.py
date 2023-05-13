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

from networkx.algorithms import community

from infectious_experiment.config_auto import EVAL_ZETA_1, EVAL_ZETA_0
from infectious_experiment.environments.rewards import InfectiousReward, calc_percent_healthy


def evaluate_auto(env, agent, num_eps, num_timesteps, seeds):

    communities_generator = community.girvan_newman(env.state.population_graph)
    communities = tuple(sorted(c) for c in next(communities_generator))
    num_communities = len(communities)
    # Map individuals in the graph to a community
    communities_map = {
        individual: comm_i for comm_i, comm in enumerate(communities) for individual in comm
    }

    num_groups = num_communities

    reward_fn = InfectiousReward()

    eval_data = {
        'tot_rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        'tot_percent_sick_over_time': np.zeros((num_eps, num_timesteps)),  # The percentage of sick per timestep per episode
        'tot_percent_healthy_over_time': np.zeros((num_eps, num_timesteps)),  # The percentage of sick per timestep per episode
        'tot_deltas_over_time': np.zeros((num_eps, num_timesteps)),  # The delta per timestep per episode
        'tot_vaccines': np.zeros((num_eps, num_groups)), # vaccine per community per timestep per episode
        'tot_infected': np.zeros((num_eps, num_groups)), # newly infected per community per timestep per episode
        'rewards': np.zeros(num_eps), # The total rewards per episode 
        'bias': np.zeros(1), # bias
    }

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        # Keep track of how many vaccines go to each community
        num_vaccines_per_community = np.zeros(num_communities)
        # Newly infected in each community
        num_newly_infected_per_community = np.zeros(num_communities)

        obs = env.reset()
        done = False

        for t in tqdm.trange(num_timesteps):
            prev_health_states = copy.deepcopy(env.state.health_states)

            action = agent.predict(obs)[0]
            a = env.process_action(action)
            obs, _, done, _ = env.step(action)

            if a is not None:
                comm_i = communities_map[np.array([a]).flatten()[0]]
                num_vaccines_per_community[comm_i] += 1
            # Compute newly infected
            for i, (health_state, prev_health_state) in enumerate(
                    zip(env.state.health_states, prev_health_states)):
                # 1 is the index in self.env.state.params.state_names for infected
                if health_state == 1 and health_state != prev_health_state:
                    comm_i = communities_map[i]
                    num_newly_infected_per_community[comm_i] += 1
            
            r = reward_fn(health_states=env.state.health_states,
                          num_vaccines_per_community=num_vaccines_per_community,
                          num_newly_infected_per_community=num_newly_infected_per_community,
                          eta0=EVAL_ZETA_0,
                          eta1=EVAL_ZETA_1)

            percent_healthy = calc_percent_healthy(env.state.health_states)
            eval_data['tot_rews_over_time'][ep][t] = r
            eval_data['tot_percent_sick_over_time'][ep][t] = 1 - percent_healthy
            eval_data['tot_percent_healthy_over_time'][ep][t] = percent_healthy
            eval_data['tot_deltas_over_time'][ep][t] = reward_fn.calc_delta(num_vaccines_per_community=num_vaccines_per_community,
                                                                            num_newly_infected_per_community=num_newly_infected_per_community)

            if done:
                break

        eval_data['tot_vaccines'][ep] = num_vaccines_per_community
        eval_data['tot_infected'][ep] = num_newly_infected_per_community

    eval_data['rewards'] = eval_data['tot_percent_healthy_over_time'].mean(axis=1)

    vaccine = eval_data['tot_vaccines']
    infect = eval_data['tot_infected']+1
    ratio = np.divide(vaccine.sum(axis=0), infect.sum(axis=0))
    eval_data['bias'] = ratio.max()-ratio.min()

    return eval_data