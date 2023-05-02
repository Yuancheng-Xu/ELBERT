from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
# from sympy import E
import torch
import tqdm

import sys
sys.path.append('..')

from lending_experiment.config_auto import NUM_GROUPS, EVAL_ZETA_0, EVAL_ZETA_1
from lending_experiment.environments.rewards import LendingReward


def evaluate_auto(env, agent, num_eps, num_timesteps, seeds):

    eval_data = {
        'tot_loans': np.zeros((num_eps, NUM_GROUPS)),  # The number of loans per group per episode
        'tot_tp': np.zeros((num_eps, NUM_GROUPS)),  # The number of true positives, or no default given loan accepted, per group per episode
        'tot_fp': np.zeros((num_eps, NUM_GROUPS)),  # The number of false positives, or default given loan accepted, per group per episode
        'tot_tn': np.zeros((num_eps, NUM_GROUPS)),  # The number of true negatives, or default given loan rejected, per group per episode
        'tot_fn': np.zeros((num_eps, NUM_GROUPS)),  # The number of false negatives, or no default given loan rejected, per group per episode
        'tot_tpr': np.zeros((num_eps, NUM_GROUPS)),  # The TPR per group per episode
        'tot_rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        'tot_loans_over_time': np.zeros((num_eps, num_timesteps,  NUM_GROUPS)),  # The number of loans per group per timestep per episode
        'tot_bank_cash_over_time': np.zeros((num_eps, num_timesteps)),  # The amount of bank cash per timestep per episode
        'tot_tp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TP per group per timestep per episode
        'tot_fp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FP per group per timestep per episode
        'tot_tn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TN per group per timestep per episode
        'tot_fn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FN per group per timestep per episode
        'tot_tpr_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TPR per group per timestep per episode
    }

    reward_fn = LendingReward()

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        obs = env.reset()
        done = False

        for t in tqdm.trange(num_timesteps):
            will_default = env.state.will_default

            action = agent.predict(obs)[0]

            group_id = np.argmax(env.state.group)
            # Add to loans if the agent wants to loan
            if action == 1:
                eval_data['tot_loans'][ep][group_id] += 1
                # Check if agent would default
                if env.state.will_default:
                    eval_data['tot_fp'][ep][group_id] += 1
                else:
                    eval_data['tot_tp'][ep][group_id] += 1
            elif action == 0:
                if env.state.will_default:
                    eval_data['tot_tn'][ep][group_id] += 1
                else:
                    eval_data['tot_fn'][ep][group_id] += 1

            # Update TPR for both groups per timestep.
            # In the edge case where denom is 0, set TPR to 0
            eval_data['tot_tpr'][ep] = np.divide(
                eval_data['tot_tp'][ep],
                eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep],
                out=np.zeros_like(eval_data['tot_tp'][ep]),
                where=(eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep])!=0)
            # Update total loans and TPR observed so far
            eval_data['tot_loans_over_time'][ep][t] = eval_data['tot_loans'][ep]
            eval_data['tot_tpr_over_time'][ep][t] = eval_data['tot_tpr'][ep]

            eval_data['tot_tp_over_time'][ep][t] = eval_data['tot_tp'][ep]
            eval_data['tot_fp_over_time'][ep][t] = eval_data['tot_fp'][ep]
            eval_data['tot_tn_over_time'][ep][t] = eval_data['tot_tn'][ep]
            eval_data['tot_fn_over_time'][ep][t] = eval_data['tot_fn'][ep]

            old_bank_cash = env.state.bank_cash

            obs, _, done, _ = env.step(action)

            bank_cash = env.state.bank_cash

            r = reward_fn(old_bank_cash=old_bank_cash,
                          bank_cash=bank_cash,
                          tpr=eval_data['tot_tpr'][ep],
                          zeta0=EVAL_ZETA_0,
                          zeta1=EVAL_ZETA_1)

            eval_data['tot_rews_over_time'][ep][t] = r
            eval_data['tot_bank_cash_over_time'][ep][t] = bank_cash

            if done:
                break

    return eval_data