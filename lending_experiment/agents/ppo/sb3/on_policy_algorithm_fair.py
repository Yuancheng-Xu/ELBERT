import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.buffers import DictRolloutBuffer  
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy      
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv # though we use our own vec_env, here it is only for type checking

# fairness specific
from agents.ppo.sb3.buffers_fair import RolloutBuffer_fair
from agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair, BasePolicy
from agents.ppo.sb3.utils_fair import evaluate_fair

# create env for evaluation 
from lending_experiment.config_fair import CLUSTER_PROBABILITIES, GROUP_0_PROB, BANK_STARTING_CASH, INTEREST_RATE, \
    CLUSTER_SHIFT_INCREMENT,EVAL_NUM_EPS,EP_TIMESTEPS_EVAL
from lending_experiment.environments.lending import DelayedImpactEnv
from lending_experiment.environments.rewards import LendingReward_fair
from lending_experiment.environments.lending_params import DelayedImpactParams, two_group_credit_clusters
from lending_experiment.agents.ppo.ppo_wrapper_env_fair import PPOEnvWrapper_fair

# for writing evaluation results to disk
import pandas as pd 
import os

# Chenghao's env
from lending_experiment.new_env import create_GeneralDelayedImpactEnv


class OnPolicyAlgorithm_fair(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.

    modification: deal with 5 rewards (using "list")
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy_fair]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,                                     # buffer_size (yes?)
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy_fair,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,

        eval_write_path: str = None,
        eval_interval: int = None, # evaluate every eval_interval times of rollout
    ):
        
        super(OnPolicyAlgorithm_fair, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        # for eval
        self.eval_write_path = eval_write_path
        self.eval_interval = eval_interval

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer_fair # original code
        buffer_cls = RolloutBuffer_fair
        # test
        if isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError('Using DictRolloutBuffer from sb3; Why? Then need to rewrite their buffer too?')
        else:
            print('Using RolloutBuffer_fair frorm agents.ppo.sb3.buffers_fair')

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable # in BaseAlgorithm, self.policy_class = policy in int(). 
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer_fair,
        n_rollout_steps: int,               # buffer size (yes?)
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        # reset env (in previous version, env is not reset)
        self._last_obs = env.reset()

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor) # values is a list of length 5
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)


            new_obs, rewards, dones, infos = env.step(clipped_actions) # rewards is a list of length 5

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                raise ValueError('on_step() is False, why?')
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        # TODO: check whether the following is correct
                        predicted_values = self.policy.predict_values(terminal_obs)
                        terminal_value = [predicted_values[i][0] for i in range(5)]
                        # terminal_value = self.policy.predict_values(terminal_obs)[0] # original code
                    for i in range(5):
                        rewards[i][idx] += self.gamma * terminal_value[i]
                    # rewards[idx] += self.gamma * terminal_value # original code

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device)) # a list of length 5

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones) 

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm_fair",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm_fair":
        '''
        Assume num_envs = 1
        total_timesteps: the total number of env.step() during the whole on policy training process (could be very large)
        self.num_timesteps: the current number of env.step() so far. 
        For each call of collect_rollouts() 
            self.num_timesteps += n_rollout_steps
            self.train() is called once
        '''
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        ) 

        callback.on_training_start(locals(), globals())

        eval_time_flag = None

        while self.num_timesteps < total_timesteps:

            ### evaluation
            if self.eval_interval is not None and (iteration) % self.eval_interval == 0:
                self.policy.set_training_mode(False)
                # create a new env for eval
                if 'General' in str(type(self.env.envs[0].env.env)):
                    # Chenghao's modifed env
                    print('Evaluation: Using Chenghao\'s modified env')
                    env_eval = create_GeneralDelayedImpactEnv()
                else:
                    print('Evaluation: Using Original Eric\'s modified env')
                    env_params = DelayedImpactParams(
                        applicant_distribution=two_group_credit_clusters(
                            cluster_probabilities=CLUSTER_PROBABILITIES,
                            group_likelihoods=[GROUP_0_PROB, 1 - GROUP_0_PROB]),
                        bank_starting_cash=BANK_STARTING_CASH,
                        interest_rate=INTEREST_RATE,
                        cluster_shift_increment=CLUSTER_SHIFT_INCREMENT,
                    )
                    env_eval = DelayedImpactEnv(env_params)

                env_eval=PPOEnvWrapper_fair(env=env_eval, reward_fn=LendingReward_fair,ep_timesteps=EP_TIMESTEPS_EVAL) # Same as Eric: EP_TIMESTEPS_EVAL = 10000, larger than EP_TIMESTEPS=2000 during training
                # evaluate and write to disk
                eval_data = evaluate_fair(env_eval, self.policy, num_eps=EVAL_NUM_EPS)
                eval_data['num_timesteps'] = self.num_timesteps
                eval_data['time_elapsed'] = str(timedelta(seconds=time.time() - eval_time_flag)) if eval_time_flag is not None else str(0)
                df_eval = pd.DataFrame([eval_data], columns=eval_data.keys())
                df_eval.to_csv(self.eval_write_path , mode='a', header=not os.path.exists(self.eval_write_path))            
            
                eval_time_flag = time.time()



            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            # check if one buffer contains exactly an interger number of episodes
            num_eps_ = (self.rollout_buffer.episode_starts==1).sum()
            for i in range(num_eps_):
                assert self.rollout_buffer.episode_starts[i * self.env.envs[0].env.ep_timesteps] == 1 

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                # self.ep_info_buffer comes from Monitor_fair
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    # ep_info["r"] is the sum of raw reward
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_mean_U_0", safe_mean([ep_info["r_U_0"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_mean_B_0", safe_mean([ep_info["r_B_0"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_mean_U_1", safe_mean([ep_info["r_U_1"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_mean_B_1", safe_mean([ep_info["r_B_1"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
    
