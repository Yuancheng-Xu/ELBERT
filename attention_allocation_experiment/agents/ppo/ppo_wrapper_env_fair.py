import gym
import numpy as np
import torch
from gym import spaces

from attention_allocation_experiment.config_fair import EP_TIMESTEPS, OBS_HIST_LEN, ZETA_0, ZETA_1
from attention_allocation_experiment.environments.rewards import AttentionAllocationReward_fair

DELTA_NOT_INCLUDE = True # default: true, not include the ratio "delta" in the obs space

class PPOEnvWrapper_fair(gym.Wrapper):
  '''
  Observation space: Observation history (of length OBS_HIST_LEN) of incidents seen, occurred, attention allocated per site
  In Eric's code, it also includes delta terms, but we put the fairness terms in fairness reward signals, not in the observation space.

  the reward (after applying this wrapper) will be of the form:
  [main_reward, [r_u_0,r_u_1,r_u_2], [r_b_0, r_b_1, r_b_2]] (assuming there are 3 groups)
  '''
  def __init__(self,
               env,
               reward_fn,
               ep_timesteps=EP_TIMESTEPS):    
    super(PPOEnvWrapper_fair, self).__init__(env)

    # It's best to turn the observations into a single vector in box space, rather than dict.
    if DELTA_NOT_INCLUDE:
      self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(env.state.params.n_locations * 3 * OBS_HIST_LEN,),
        dtype=np.float32
      )
    else:
      self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(env.state.params.n_locations * 4 * OBS_HIST_LEN,),
        dtype=np.float32
      )

    self.action_space = spaces.Box(
      low=-3,
      high=3,
      shape=(env.state.params.n_locations,),
      dtype=np.float64
    )

    self.env = env
    self.reward_fn = reward_fn()

    self.ep_timesteps = ep_timesteps
    self.timestep = 0

    self.ep_incidents_seen = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
    self.ep_incidents_occurred = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))


    # Observation history of incidents seen, occurred, attention allocated per site
    if DELTA_NOT_INCLUDE:
      self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 3))
    else:
      self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 4))

  def reset(self):
    self.timestep = 0
    self.ep_incidents_seen = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
    self.ep_incidents_occurred = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
    if DELTA_NOT_INCLUDE:
      self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 3))
    else:
      self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * 4))

    _ = self.env.reset()

    return self.observation_history.flatten()

  def step(self, action):
    action = self.process_action(action)
    obs, _, done, info = self.env.step(action)

    self.ep_incidents_seen[self.timestep] = self.env.state.incidents_seen
    self.ep_incidents_occurred[self.timestep] = self.env.state.incidents_occurred

    self.timestep += 1

    if self.timestep >= self.ep_timesteps:
      done = True

    # Form observation history: pop the oldest and append the current observation
    if DELTA_NOT_INCLUDE:
      current_obs = np.concatenate((self.env.state.incidents_seen, self.env.state.incidents_occurred, action), axis=0)
    else:
       deltas = np.sum(self.ep_incidents_seen, axis=0) / (np.sum(self.ep_incidents_occurred, axis=0) + 1)
       current_obs = np.concatenate((self.env.state.incidents_seen, self.env.state.incidents_occurred, action, deltas), axis=0)

    self.observation_history = np.concatenate((self.observation_history[1:], np.expand_dims(current_obs, axis=0)))
    obs = self.observation_history.flatten()

    # Store custom reward
    reward = self.reward_fn(incidents_seen=self.env.state.incidents_seen,
                            incidents_occurred=self.env.state.incidents_occurred,
                            zeta0=ZETA_0,
                            zeta1=ZETA_1)
    
    # Fairness signals r_U and r_B (type: numpy.ndarray of shape = (num_groups,))
    r_U = self.env.state.incidents_seen
    r_B = self.env.state.incidents_occurred 

    return obs, [reward,r_U,r_B], done, info

  def process_action(self, action):
    """
    Convert actions from logits to attention allocation over sites through the following steps:
    1. Convert logits vector into a probability distribution using softmax
    2. Convert probability distribution into allocation distribution with multinomial distribution

    Args:
      action: n_locations vector of logits

    Returns: n_locations vector of allocations
    """
    action = torch.tensor(action)
    probs = torch.nn.functional.softmax(action, dim=-1)
    allocs = [0 for _ in range(self.env.state.params.n_locations)]
    n_left = self.env.state.params.n_attention_units
    while n_left > 0:
      ind = np.argmax(probs)
      allocs[ind] += 1
      n_left -= 1
      probs[ind] -= 1 / self.env.state.params.n_attention_units

    action = np.array(allocs)
    return action
