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
from attention_allocation_experiment.config_auto import N_LOCATIONS, INCIDENT_RATES, N_ATTENTION_UNITS, DYNAMIC_RATE, EVAL_ZETA_0, EVAL_ZETA_1, EVAL_ZETA_2, \
    SAVE_DIR, EXP_DIR, POLICY_KWARGS, LEARNING_RATE, SAVE_FREQ, TRAIN_TIMESTEPS, EVAL_MODEL_PATHS, CPO_EVAL_MODEL_PATHS
from attention_allocation_experiment.config import OBS_HIST_LEN
from environments.attention_allocation import LocationAllocationEnv, Params
from environments.rewards import AttentionAllocationReward


def evaluate_auto(env, agent, num_eps, num_timesteps, seeds):

    reward_fn = AttentionAllocationReward()

    eval_data = {
        'tot_rews': np.zeros((num_eps, num_timesteps)),  # The rewards per timestep per episode
        'tot_att_all': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The attention allocated per site per timestep per episode
        'tot_true_rates': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The true rates per site per timestep per episode
        'tot_deltas': np.zeros((num_eps, num_timesteps)),  # The deltas per timestep per episode
        'tot_incidents_seen': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The incidents seen per site per timestep per episode
        'tot_incidents_occurred': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The incidents occurred per site per timestep per episode
        'tot_incidents_missed': np.zeros((num_eps, num_timesteps, env.state.params.n_locations)),  # The incidents missed per site per timestep per episode
        'tot_rew_infos': []  # The values of each term in the reward per timestep per episode, shape is (num_eps, num_timesteps, dict)
    }

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        ep_data = {
            'rews': np.zeros(num_timesteps),  # The reward per timestep of this episode
            'att_all': np.zeros((num_timesteps, env.state.params.n_locations)),  # The attention allocated per site per timestep of this episode
            'true_rates': np.zeros((num_timesteps, env.state.params.n_locations)),  # The true rates per site per timestep of this episode
            'deltas': np.zeros(num_timesteps),  # The deltas per timestep of this episode
            'ep_incidents_seen': np.zeros((num_timesteps, env.state.params.n_locations)),  # The incidents seen per site per timestep of this episode
            'ep_incidents_occurred': np.zeros((num_timesteps, env.state.params.n_locations)),  # The incidents occurred per site per timestep of this episode
            'rew_infos': [],  # The values of each term in the reward per timestep of this episode
        }

        obs = env.reset()
        done = False

        for t in tqdm.trange(num_timesteps):
            action = agent.predict(obs)[0]
            obs, _, done, _ = env.step(action)

            # Update total incidents variables
            ep_data['ep_incidents_seen'][t] = env.state.incidents_seen
            ep_data['ep_incidents_occurred'][t] = env.state.incidents_occurred

            r = reward_fn(incidents_seen=env.state.incidents_seen,
                          incidents_occurred=env.state.incidents_occurred,
                          ep_incidents_seen=ep_data['ep_incidents_seen'],
                          ep_incidents_occurred=ep_data['ep_incidents_occurred'],
                          zeta0=EVAL_ZETA_0,
                          zeta1=EVAL_ZETA_1,
                          zeta2=EVAL_ZETA_2,
                          )

            ep_data['rew_infos'].append(reward_fn.rew_info)
            ep_data['rews'][t] = r
            ep_data['att_all'][t] = env.process_action(action)
            ep_data['true_rates'][t] = env.state.params.incident_rates
            ep_data['deltas'][t] = reward_fn.calc_delta(ep_incidents_seen=ep_data['ep_incidents_seen'],
                                                        ep_incidents_occurred=ep_data['ep_incidents_occurred'])

            if done:
                break

        # Store the episodic data in eval data
        eval_data['tot_rews'][ep] = ep_data['rews']
        eval_data['tot_att_all'][ep] = ep_data['att_all']
        eval_data['tot_true_rates'][ep] = ep_data['true_rates']
        eval_data['tot_deltas'][ep] = ep_data['deltas']
        eval_data['tot_incidents_seen'][ep] = ep_data['ep_incidents_seen']
        eval_data['tot_incidents_occurred'][ep] = ep_data['ep_incidents_occurred']
        eval_data['tot_incidents_missed'][ep] = ep_data['ep_incidents_occurred'] - ep_data['ep_incidents_seen']
        eval_data['tot_rew_infos'].append(copy.deepcopy(ep_data['rew_infos']))

    return eval_data