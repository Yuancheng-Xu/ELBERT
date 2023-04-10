import gym
import numpy as np
import torch
from gym import spaces

from lending_experiment.config_fair import EP_TIMESTEPS


class PPOEnvWrapper_fair(gym.Wrapper):
  def __init__(self,
               env,
               reward_fn,
               ep_timesteps=EP_TIMESTEPS):
    '''
    ep_timesteps: done=True if rollout for more than ep_timesteps steps
    reward_fn should be LendingReward_fair in rewards.py
    '''
    super(PPOEnvWrapper_fair, self).__init__(env)

    self.observation_space = spaces.Box(
      low=np.inf,
      high=np.inf,
      # (7) OHE of credit score + (2) group 
      shape=(env.observation_space['applicant_features'].shape[0] + 1 * env.state.params.num_groups,),

      # Eric:
      # (7) OHE of credit score + (2) group +  (2) TPRs of each group
      # shape=(env.observation_space['applicant_features'].shape[0] + 2 * env.state.params.num_groups,),
    )

    self.action_space = spaces.Discrete(n=2)

    self.env = env
    self.reward_fn = reward_fn()

    self.timestep = 0
    self.ep_timesteps = ep_timesteps

    self.old_bank_cash = 0

    # Eric:
    # self.tp = np.zeros(self.env.state.params.num_groups,)
    # self.fp = np.zeros(self.env.state.params.num_groups,)
    # self.tn = np.zeros(self.env.state.params.num_groups,)
    # self.fn = np.zeros(self.env.state.params.num_groups,)
    # self.tpr = np.zeros(self.env.state.params.num_groups,)
    # self.delta = np.zeros(1, )
    # self.delta_delta = 0

  def process_observation(self, obs):
    '''
    Input obs contains all "self.observable_state_vars" of the env in lending.py
    Including: 'bank_cash', 'applicant_features', 'group'
    '''
    credit_score = obs['applicant_features']
    group = obs['group']

    return np.concatenate(
      (credit_score,
       group,
      #  self.tpr, # Eric
       ),
      axis=0
    )

  
  # def compute_tpr(self, tp, fn):
  #   # Eric
  #   # tp: true positive, 2-dimensional for 2 groups
  #   # fn: false negative, 2-dimensional for 2 groups
  #   return np.divide(
  #     tp,
  #     tp + fn,
  #     out=np.zeros_like(tp),
  #     where=(tp + fn) != 0)

  def reset(self):
    self.timestep = 0
    self.old_bank_cash = 0

    # Eric
    # self.tp = np.zeros(self.env.state.params.num_groups,)
    # self.fp = np.zeros(self.env.state.params.num_groups,)
    # self.tn = np.zeros(self.env.state.params.num_groups,)
    # self.fn = np.zeros(self.env.state.params.num_groups,)
    # self.tpr = np.zeros(self.env.state.params.num_groups,)
    # self.delta = np.zeros(1, )
    # self.delta_delta = 0

    return self.process_observation(self.env.reset())

  def step(self, action):
    '''
    modify the reward to 5 dimension: r = (agent reward, r_U_0, r_B_0, r_U_1, r_B_1)
    r_U_g = 1 if group = g and not default and action = 1, = 0 otherwise
    r_B_g = 1 if group = g and not default               , = 0 otherwise
    '''

    # Fairness reward
    r_U_0 = 0
    r_B_0 = 0
    r_U_1 = 0
    r_B_1 = 0

    # Update fairness reward before we step the environment
    group_id = np.argmax(self.env.state.group)

    if group_id == 0:
      if not self.env.state.will_default:
        r_B_0 = 1
        if action == 1:
          r_U_0 = 1
    elif group_id == 1:
      if not self.env.state.will_default:
        r_B_1 = 1
        if action == 1:
          r_U_1 = 1
    else:
      raise ValueError('invalid group_id')
    
    # only for test, can be removed
    assert r_B_0 * r_B_1 == 0, 'r_B for both group are non-zero'
    assert (r_B_0 >= r_U_0) and (r_B_1 >= r_U_1), 'r_B < r_U error!'    

    self.old_bank_cash = self.env.state.bank_cash

    obs, _, done, info = self.env.step(action) # not using the reward giving by env

    r = self.reward_fn(old_bank_cash=self.old_bank_cash,
                       bank_cash=self.env.state.bank_cash)

    self.timestep += 1
    if self.timestep >= self.ep_timesteps:
      done = True

    return self.process_observation(obs), [r,r_U_0, r_B_0, r_U_1, r_B_1], done, info
