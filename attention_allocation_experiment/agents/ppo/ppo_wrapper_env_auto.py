import gym
import numpy as np
import torch
from gym import spaces

from attention_allocation_experiment.config_auto import EP_TIMESTEPS, OBS_HIST_LEN, ZETA_0, ZETA_1, ZETA_2, EVAL_ZETA_0, EVAL_ZETA_1, EVAL_ZETA_2

DELTA_OBS = False # True: include delta in obs, as in Eric's code

class PPOEnvWrapper_auto(gym.Wrapper):
  def __init__(self,
               env,
               reward_fn,
               name,
               ep_timesteps=EP_TIMESTEPS,
               delta_obs=False):
    super(PPOEnvWrapper_auto, self).__init__(env)

    # It's best to turn the observations into a single vector in box space, rather than dict.
    self.delta_obs = delta_obs
    self.obs_size = 3+int(delta_obs)
    self.observation_space = spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(env.state.params.n_locations * self.obs_size * OBS_HIST_LEN,),
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


    # Observation history of incidents seen, occurred, attention allocated, delta terms per site
    self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * self.obs_size))

    self.delta = 0  # The delta term
    self.delta_delta = 0  # The delta(s') - delta(s) part of the decrease-in-violation term of Eq. 3 from the paper, where s' is the consecutive state of s
    self.zeta0 = EVAL_ZETA_0 if name=="R-PPO" else ZETA_0
    self.zeta1 = EVAL_ZETA_1 if name=="R-PPO" else ZETA_1
    self.zeta2 = EVAL_ZETA_2 if name=="R-PPO" else ZETA_2

  def reset(self):
    self.timestep = 0
    self.ep_incidents_seen = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
    self.ep_incidents_occurred = np.zeros((self.ep_timesteps, self.env.state.params.n_locations))
    self.observation_history = np.zeros((OBS_HIST_LEN, self.env.state.params.n_locations * self.obs_size))
    self.delta = 0
    self.delta_delta = 0

    _ = self.env.reset()

    return self.observation_history.flatten()

  def step(self, action):
    action = self.process_action(action)
    obs, reward, done, info = self.env.step(action)

    self.ep_incidents_seen[self.timestep] = self.env.state.incidents_seen
    self.ep_incidents_occurred[self.timestep] = self.env.state.incidents_occurred

    self.timestep += 1

    if self.timestep >= self.ep_timesteps:
      done = True

    # Form observation history
    if self.delta_obs:
      deltas = np.sum(self.ep_incidents_seen, axis=0) / (np.sum(self.ep_incidents_occurred, axis=0) + 1)
      current_obs = np.concatenate((self.env.state.incidents_seen, self.env.state.incidents_occurred, action, deltas), axis=0)
    else:
      current_obs = np.concatenate((self.env.state.incidents_seen, self.env.state.incidents_occurred, action), axis=0)
    self.observation_history = np.concatenate((self.observation_history[1:], np.expand_dims(current_obs, axis=0)))
    obs = self.observation_history.flatten()

    # Store custom reward
    reward = self.reward_fn(incidents_seen=self.env.state.incidents_seen,
                            incidents_occurred=self.env.state.incidents_occurred,
                            ep_incidents_seen=self.ep_incidents_seen,
                            ep_incidents_occurred=self.ep_incidents_occurred,
                            zeta0=self.zeta0,
                            zeta1=self.zeta1,
                            zeta2=self.zeta2
                          )

    # Update delta terms
    old_delta = self.delta
    self.delta = self.reward_fn.calc_delta(self.ep_incidents_seen, self.ep_incidents_occurred)
    self.delta_delta = self.delta - old_delta

    return obs, reward, done, info

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
