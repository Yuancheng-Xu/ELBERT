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
from infectious_experiment.agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair
from infectious_experiment.agents.ppo.ppo_wrapper_env_fair import PPOEnvWrapper_fair
from infectious_experiment.environments.rewards import InfectiousReward_fair,calc_percent_healthy
import numpy as np
import torch
from networkx.algorithms import community
import random
import copy

class DummyVecEnv_fair(DummyVecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super().__init__(env_fns)
        
        self.num_groups = env_fns[0]().env.num_communities
        self.buf_rews = [np.zeros((self.num_envs,), dtype=np.float32),[np.zeros((self.num_envs,), dtype=np.float32) for g in range(self.num_groups)],[np.zeros((self.num_envs,), dtype=np.float32) for g in range(self.num_groups)]]

    def step_wait(self) -> Tuple[VecEnvObs, List[np.ndarray], np.ndarray, List[Dict]]:
        for env_idx in range(self.num_envs):
            obs, buf_rews_list_env_idx, self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            self.buf_rews[0][env_idx] = buf_rews_list_env_idx[0]
            for g in range(self.num_groups):
                self.buf_rews[1][g][env_idx] = buf_rews_list_env_idx[1][g]
                self.buf_rews[2][g][env_idx] = buf_rews_list_env_idx[2][g]

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), copy.deepcopy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
    
class Monitor_fair(Monitor):
    def __init__(self, env: gym.Env, filename: Optional[str] = None, allow_early_resets: bool = True, reset_keywords: Tuple[str, ...] = (), info_keywords: Tuple[str, ...] = ()):
        super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)

        self.num_groups =  env.num_communities
        # e.g.: [ [], [[],[],[]], [[],[],[]] ]
        self.rewards: List[Union[List[float],List[List[float]]]] = [[],[[] for g in range(self.num_groups)], [[] for g in range(self.num_groups)]] 
        self.episode_returns: List[Union[List[float],List[List[float]]]] = [[],[[] for g in range(self.num_groups)], [[] for g in range(self.num_groups)]]
    
    def reset(self, **kwargs) -> GymObs:
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = [[],[[] for g in range(self.num_groups)], [[] for g in range(self.num_groups)]] 
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

        self.rewards[0].append(reward[0])
        for g in range(self.num_groups):
            self.rewards[1][g].append(reward[1][g])
            self.rewards[2][g].append(reward[2][g])

        if done:
            self.needs_reset = True
            ep_rew = [sum(self.rewards[0]), [sum(self.rewards[1][g]) for g in range(self.num_groups)], [sum(self.rewards[2][g]) for g in range(self.num_groups)]]
            ep_len = len(self.rewards[0])
            assert ep_len == len(self.rewards[1][0]), 'reward lengths are different'
            # ep_info = {"r": round(ep_rew[0], 6), "r_U_0": round(ep_rew[1], 6), "r_B_0": round(ep_rew[2], 6), "r_U_1": round(ep_rew[3], 6), \
            #            "r_B_1": round(ep_rew[4], 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            ep_info = {"r": round(ep_rew[0], 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]

            self.episode_returns[0].append(ep_rew[0])
            for g in range(self.num_groups):
                self.episode_returns[1][g].append(ep_rew[1][g])
                self.episode_returns[2][g].append(ep_rew[2][g])
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
    env: should be the one with fairness reward signals
    num_eps: number of episodes
    write_path: add data recording to path

    write to disk the evaluation results of reward and bias
    '''
    assert isinstance(agent, ActorCriticPolicy_fair), 'evaluate_fair only works for ActorCriticPolicy_fair policy'
    assert isinstance(env, PPOEnvWrapper_fair), 'env should be of type: PPOEnvWrapper_fair'

    num_groups = env.num_communities
    seeds = [random.randint(0, 10000) for _ in range(num_eps)]
    num_timesteps = env.ep_timesteps # number of steps per episodes (unless done=True) 

    reward_fn = InfectiousReward_fair()

    # helper variables
    eval_data = {
        'rews_over_time': np.zeros((num_eps, num_timesteps)),  # The reward per timestep per episode
        'vaccines_over_time': np.zeros((num_eps, num_timesteps, num_groups)), # vaccine per community per timestep per episode
        'infected_over_time': np.zeros((num_eps, num_timesteps, num_groups)), # newly infected per community per timestep per episode
    }

    communities_generator = community.girvan_newman(env.state.population_graph)
    communities = tuple(sorted(c) for c in next(communities_generator))
    # Map individuals in the graph to a community
    communities_map = {
        individual: comm_i for comm_i, comm in enumerate(communities) for individual in comm
    }

    for ep in range(num_eps):
        random.seed(seeds[ep])
        np.random.seed(seeds[ep])
        torch.manual_seed(seeds[ep])

        obs = env.reset()
        done = False

        for t in range(num_timesteps):

            prev_health_states = copy.deepcopy(env.state.health_states)

            action = None
            action = agent.predict(obs)[0]
            a = env.process_action(action)

            obs, _, done, _ = env.step(action)

            # Update the number of vaccines in each community
            if a is not None:
                comm_i = communities_map[np.array([a]).flatten()[0]]
                eval_data['vaccines_over_time'][ep][t][comm_i] += 1
            # Compute newly infected
            for i, (health_state, prev_health_state) in enumerate(
                    zip(env.state.health_states, prev_health_states)):
                # 1 is the index in self.env.state.params.state_names for infected
                if health_state == 1 and health_state != prev_health_state:
                    comm_i = communities_map[i]
                    eval_data['infected_over_time'][ep][t][comm_i] += 1

            r = reward_fn(health_states=env.state.health_states)
            eval_data['rews_over_time'][ep][t] = r
            
            if done:
                break

    U = np.sum(eval_data['vaccines_over_time'],axis=(0,1))
    B = np.sum(eval_data['infected_over_time'],axis=(0,1)) + 1 * num_eps # 1 * num_eps is for adaptation to the formula in Eric's paper
    # essential (only write these to disk): average across episodes and timesteps
    eval_data_essential = {}
    eval_data_essential['return'] = eval_data['rews_over_time'].mean() # average across episodes and timesteps
    ratio_list = []
    for g in range(num_groups):
        eval_data_essential['ratio_{}'.format(g)] = U[g]/B[g]
        ratio_list.append(U[g]/B[g])
    eval_data_essential['bias'] = max(ratio_list) - min(ratio_list)
    eval_data_essential['benefit_max'] = max(ratio_list)
    eval_data_essential['benefit_min'] = min(ratio_list)


    return eval_data_essential

