'''
1. Modify the sb3 dummy_vec_env to deal with multiple rewards
original code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/dummy_vec_env.py

2. Modify the sb3 Monitor to deal with multiple rewards
original code: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/monitor.py

3. evaluation
evaluate the model during training (instead of saving checkpoints as done in Eric's code)
'''
import time

from stable_baselines3.common.vec_env.dummy_vec_env import *
from typing import Dict, List, Tuple

from stable_baselines3.common.monitor import * 
from stable_baselines3.common.type_aliases import GymObs

# for evaluation
from lending_experiment.agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair
import numpy as np
import torch
# import tqdm
import random

class DummyVecEnv_fair(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super().__init__(env_fns)

        self.buf_rews = [np.zeros((self.num_envs,), dtype=np.float32) for i in range(5)]

    def step_wait(self) -> Tuple[VecEnvObs, List[np.ndarray], np.ndarray, List[Dict]]:
        for env_idx in range(self.num_envs):
            obs, buf_rews_list_env_idx, self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            for i in range(5):
                  self.buf_rews[i][env_idx] = buf_rews_list_env_idx[i]
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
    
class Monitor_fair(Monitor):
    def __init__(self, env: gym.Env, filename: Optional[str] = None, allow_early_resets: bool = True, reset_keywords: Tuple[str, ...] = (), info_keywords: Tuple[str, ...] = ()):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)

        self.rewards: List[List[float]] = [[] for i in range(5)]
        self.episode_returns: List[List[float]] = [[] for i in range(5)]
    
    def reset(self, **kwargs) -> GymObs:
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = [[] for i in range(5)]
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)
    
    def step(self, action: Union[np.ndarray, int]) -> Tuple[GymObs, List[float], bool, Dict]:
        """
        Step the environment with the given action
        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        for i in range(5):
            self.rewards[i].append(reward[i])
        if done:
            self.needs_reset = True
            ep_rew = [sum(self.rewards[i]) for i in range(5)]
            ep_len = len(self.rewards[0])
            assert ep_len == len(self.rewards[1]), 'reward lengths are different'
            ep_info = {"r": round(ep_rew[0], 6), "r_U_0": round(ep_rew[1], 6), "r_B_0": round(ep_rew[2], 6), "r_U_1": round(ep_rew[3], 6), \
                       "r_B_1": round(ep_rew[4], 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            for i in range(5):
                self.episode_returns[i].append(ep_rew[i])
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info
    
    def get_episode_rewards(self) -> List[List[float]]:
        """
        Returns the rewards of all the episodes
        :return:
        """
        return self.episode_returns
 

def evaluate_fair(env, agent, num_eps):
    '''
    env: should be the one with five rewards
    num_eps: number of episodes
    write_path: add data recording to path

    write to disk the evaluation results 
    1. reward (or bank cash)
    2. TPR of two groups

    TODO: need to do sanity check: whether r_U is equal to some function of the original evaluation
    '''
    assert isinstance(agent, ActorCriticPolicy_fair), 'evaluate_fair only works for ActorCriticPolicy_fair policy'

    NUM_GROUPS = 2
    seeds = [random.randint(0, 10000) for _ in range(num_eps)]
    num_timesteps = env.ep_timesteps # number of steps per episodes (unless done=True) 

    eval_data = {
        # essential (only write these to disk)
        'cash': 0,
        'tpr_0': -1,
        'tpr_1': -1,
        # helper variables
        'cash_over_time': np.zeros((num_eps, num_timesteps)), 
        'r_U': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  
        'r_B': np.zeros((num_eps, num_timesteps, NUM_GROUPS)), 

        # old (can be removed later)
        # 'tot_loans': np.zeros((num_eps, NUM_GROUPS)),  # The number of loans per group per episode
        # 'tot_tp': np.zeros((num_eps, NUM_GROUPS)),  # The number of true positives, or no default given loan accepted, per group per episode
        # 'tot_fp': np.zeros((num_eps, NUM_GROUPS)),  # The number of false positives, or default given loan accepted, per group per episode
        # 'tot_tn': np.zeros((num_eps, NUM_GROUPS)),  # The number of true negatives, or default given loan rejected, per group per episode
        # 'tot_fn': np.zeros((num_eps, NUM_GROUPS)),  # The number of false negatives, or no default given loan rejected, per group per episode
        # 'tot_tpr': np.zeros((num_eps, NUM_GROUPS)),  # The TPR per group per episode
        # 'tot_rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        # 'tot_loans_over_time': np.zeros((num_eps, num_timesteps,  NUM_GROUPS)),  # The number of loans per group per timestep per episode
        # 'tot_bank_cash_over_time': np.zeros((num_eps, num_timesteps)),  # The amount of bank cash per timestep per episode
        # 'tot_tp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TP per group per timestep per episode
        # 'tot_fp_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FP per group per timestep per episode
        # 'tot_tn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TN per group per timestep per episode
        # 'tot_fn_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The FN per group per timestep per episode
        # 'tot_tpr_over_time': np.zeros((num_eps, num_timesteps, NUM_GROUPS)),  # The TPR per group per timestep per episode
    }

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        obs = env.reset()
        done = False
        # print(f'Evaluation:  Episode {ep}:')
        # for t in tqdm.trange(num_timesteps):
        for t in range(num_timesteps):
            action = agent.predict(obs)[0]

            #################################################### sanity check begins
            # group_id = np.argmax(env.state.group)
            # r_u, r_b = [0,0],[0,0] # for sanity check
            # # Add to loans if the agent wants to loan
            # if action == 1:
            #     eval_data['tot_loans'][ep][group_id] += 1
            #     # Check if agent would default
            #     if env.state.will_default:
            #         eval_data['tot_fp'][ep][group_id] += 1
            #     else:
            #         eval_data['tot_tp'][ep][group_id] += 1
            #         r_u[group_id] = 1
            #         r_b[group_id] = 1
            # elif action == 0:
            #     if env.state.will_default:
            #         eval_data['tot_tn'][ep][group_id] += 1
            #     else:
            #         eval_data['tot_fn'][ep][group_id] += 1
            #         r_b[group_id] = 1

            # # Update TPR for both groups per timestep.
            # # In the edge case where denom is 0, set TPR to 0
            # eval_data['tot_tpr'][ep] = np.divide(
            #     eval_data['tot_tp'][ep],
            #     eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep],
            #     out=np.zeros_like(eval_data['tot_tp'][ep]),
            #     where=(eval_data['tot_tp'][ep] + eval_data['tot_fn'][ep])!=0)
            # # Update total loans and TPR observed so far
            # eval_data['tot_loans_over_time'][ep][t] = eval_data['tot_loans'][ep]
            # eval_data['tot_tpr_over_time'][ep][t] = eval_data['tot_tpr'][ep]

            # eval_data['tot_tp_over_time'][ep][t] = eval_data['tot_tp'][ep]
            # eval_data['tot_fp_over_time'][ep][t] = eval_data['tot_fp'][ep]
            # eval_data['tot_tn_over_time'][ep][t] = eval_data['tot_tn'][ep]
            # eval_data['tot_fn_over_time'][ep][t] = eval_data['tot_fn'][ep]

            # old_bank_cash = env.state.bank_cash
            #################################################### sanity check ends

            obs, rewards, done, infos = env.step(action)      ########### Inside sanity check, only this line should not be removed!

            #################################################### sanity check begins
            # bank_cash = env.state.bank_cash
            # eval_data['tot_bank_cash_over_time'][ep][t] = bank_cash

            #################################################### sanity check ends

            # essential
            eval_data['cash_over_time'][ep][t] = env.state.bank_cash
            _, eval_data['r_U'][ep][t][0], eval_data['r_B'][ep][t][0], eval_data['r_U'][ep][t][1], eval_data['r_B'][ep][t][1] = rewards

            #################################################### sanity check begins
            # assert eval_data['r_U'][ep][t][0] == r_u[0]
            # assert eval_data['r_B'][ep][t][0] == r_b[0]
            # assert eval_data['r_U'][ep][t][1] == r_u[1]
            # assert eval_data['r_B'][ep][t][1] == r_b[1]
            # assert bank_cash == eval_data['cash_over_time'][ep][t]
            #################################################### sanity check ends

            if done:
                break

    # essential 
    U = np.sum(eval_data['r_U'],axis=(0,1))
    B = np.sum(eval_data['r_B'],axis=(0,1))
    # print('U: should be just two number array',U)
    # print('B',B)
    eval_data['tpr_0'] = U[0]/B[0]
    eval_data['tpr_1'] = U[1]/B[1]
    eval_data['bias'] = eval_data['tpr_0'] - eval_data['tpr_1']
    eval_data['cash'] = eval_data['cash_over_time'][:,-1].mean()

    # delete helper variables
    eval_data.pop('r_U')
    eval_data.pop('r_B')
    eval_data.pop('cash_over_time')


    #################################################### sanity check begins
    # assert eval_data['tot_bank_cash_over_time'][:,-1].mean() == eval_data['cash']
    # print('Eric: TPR_0 with many episodes: ', eval_data['tot_tpr'][:,0])
    # print('Eric: TPR_1 with many episodes: ', eval_data['tot_tpr'][:,1])
    # print('Our TPR_0:{}, TPR_1:{}'.format(eval_data['tpr_0'],eval_data['tpr_1']))
    #################################################### sanity check ends

    return eval_data