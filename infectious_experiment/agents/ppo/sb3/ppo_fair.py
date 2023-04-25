import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch as th
from gym import spaces
from torch.nn import functional as F
import copy

            
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from infectious_experiment.agents.ppo.sb3.policies_fair import ActorCriticPolicy_fair
from infectious_experiment.agents.ppo.sb3.on_policy_algorithm_fair import OnPolicyAlgorithm_fair

from infectious_experiment.config_fair import BUFFER_SIZE_TRAINING

class PPO_fair(OnPolicyAlgorithm_fair):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
   
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance

    Modification
    1. deal with 2M + 1 rewards
    2. The loss function contains several components
        a. policy gradient for the main reward
        b. Bellman loss for all value functions
        c. a mitigation loss (the gradient of this part is consistent with our derivation)
    """

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy_fair]],    
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = BUFFER_SIZE_TRAINING, # buffer size; original: 2048
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,

            
            bias_coef: float = None,  # for mitigation
            beta_smooth: float = None, # for computing the soft bias
            
            eval_write_path: str = None,
            eval_interval: int = None, # evaluate every eval_interval times of rollout
    ):

        super(PPO_fair, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),

            eval_write_path = eval_write_path,
            eval_interval= eval_interval,
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                    batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.bias_coef = bias_coef
        self.beta_smooth = beta_smooth

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO_fair, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # for logs
        entropy_losses = []
        pg_losses, value_losses = [[], [[] for i in range(self.num_groups)], [[] for i in range(self.num_groups)]], [[], [[] for i in range(self.num_groups)], [[] for i in range(self.num_groups)]]
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = [] # for log
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions) # values is a "Fairness List"

                values[0] = values[0].flatten()
                values[1] = [i.flatten() for i in values[1]]
                values[2] = [i.flatten() for i in values[2]]
        
                # Advantages shape: (batch_size,)
                advantages = rollout_data.advantages # a "Fairness List"

                # Normalize advantage
                if self.normalize_advantage:
                    advantages[0] = (advantages[0] - advantages[0].mean()) / (advantages[0].std() + 1e-8)
                    for g in range(self.num_groups):
                        advantages[1][g] = (advantages[1][g] - advantages[1][g].mean()) / (advantages[1][g].std() + 1e-8)
                        advantages[2][g] = (advantages[2][g] - advantages[2][g].mean()) / (advantages[2][g].std() + 1e-8)


                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss (for all returns): e.g. the gradient of policy_loss[1][g] is the policy gradient of return_U_g
                policy_loss_1, policy_loss_2, policy_loss = [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]],\
                    [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]],\
                    [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]]
                for i in range(3):
                    if i == 0:
                        policy_loss_1[i] = advantages[i] * ratio
                        policy_loss_2[i] = advantages[i] * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                        policy_loss[i] = -th.min(policy_loss_1[i], policy_loss_2[i]).mean()
                        # Logging 
                        pg_losses[i].append(policy_loss[i].item())
                    else:
                        for g in range(self.num_groups):
                            policy_loss_1[i][g] = advantages[i][g] * ratio
                            policy_loss_2[i][g] = advantages[i][g] * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                            policy_loss[i][g] = -th.min(policy_loss_1[i][g], policy_loss_2[i][g]).mean()
                            # Logging 
                            pg_losses[i][g].append(policy_loss[i][g].item())

                # Logging
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    # old_values is in type_aliases.RolloutBufferSamples_fair, meaning the current value estimate
                    values_pred = [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]]
                    for i in range(3):
                        if i == 0:
                            values_pred[i] = rollout_data.old_values[i] + th.clamp(
                                values[i] - rollout_data.old_values[i], -clip_range_vf, clip_range_vf
                            )
                        else:
                            for g in range(self.num_groups):
                                values_pred[i][g] = rollout_data.old_values[i][g] + th.clamp(
                                    values[i][g] - rollout_data.old_values[i][g], -clip_range_vf, clip_range_vf
                                )

                # Value loss using the TD(gae_lambda) target, for 5 rewards
                value_loss = [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]]
                for i in range(3):
                    if i == 0:
                        value_loss[i] = F.mse_loss(rollout_data.returns[i], values_pred[i])
                        value_losses[i].append(value_loss[i].item())
                    else:
                        for g in range(self.num_groups):
                            value_loss[i][g] = F.mse_loss(rollout_data.returns[i][g], values_pred[i][g])
                            value_losses[i][g].append(value_loss[i][g].item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                ######## Loss function: the gradient of loss should be consistent with the derivation via chain rule

                # Estimate for fairness return signals: Use the TD lambda return of the first state in each episode (buffer contain several episodes)
                # Note: use the whole buffer (not minibatch); 
                # since rollout_buffer.returns does not change during one call of train(), these estimate will be the same in every for loop
                # Note: When using gae_lambda = 1 in the buffer, the following are Monte Carlo Estimate
                value_U_estimate = np.zeros(self.num_groups)   
                value_B_estimate = np.zeros(self.num_groups)  
                for g in range(self.num_groups):
                     value_U_estimate[g] = (self.rollout_buffer.returns[1][g][self.rollout_buffer.episode_starts==1]).mean()
                     value_B_estimate[g] = (self.rollout_buffer.returns[2][g][self.rollout_buffer.episode_starts==1]).mean()
                
                assert len(self.rollout_buffer.returns) == 3 # xyc test

                value_U_estimate = torch.from_numpy(value_U_estimate)
                value_B_estimate = torch.from_numpy(value_B_estimate)
                ratio_fairness = value_U_estimate / value_B_estimate

                soft_bias, soft_bias_grad = soft_bias_value_and_gradient(copy.deepcopy(ratio_fairness),self.beta_smooth)

                # estimate the gradient of R_U / R_B
                policy_loss_U = torch.zeros(self.num_groups)   
                policy_loss_B = torch.zeros(self.num_groups)  
                for g in range(self.num_groups):
                    policy_loss_U[g] = policy_loss[1][g]
                    policy_loss_B[g] = policy_loss[2][g]
                policy_loss = policy_loss[0]

                # grad of R_U/R_B w.r.t. pi
                grad_ratio_U_B = policy_loss_U / value_B_estimate - policy_loss_B * (value_U_estimate/(value_B_estimate**2))
                
                bias_loss = 2 * soft_bias * torch.sum(grad_ratio_U_B * soft_bias_grad) * (- self.bias_coef)

                # value loss for all rewards
                value_loss = value_loss[0] + sum(value_loss[1]) + sum(value_loss[2])

                # final loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + bias_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values[0].flatten(), self.rollout_buffer.returns[0].flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))

        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses[0])) 

        self.logger.record("train/value_loss", np.mean(value_losses[0])) 
        self.logger.record("train/value_loss_U", np.array([np.mean(array_g) for array_g in value_losses[1]]).mean()) # loss of value_U average acrossed all groups
        self.logger.record("train/value_loss_B", np.array([np.mean(array_g) for array_g in value_losses[2]]).mean()) 

        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())

        self.logger.record("train/explained_variance", explained_var)

        # from the last buffer (so not the current bias per se)
        # value_U_0_estimate is TD lambda return estimate (not sum of raw reward as in "ep_rew_mean_U_0" in the log)
        self.logger.record("rollout/soft_bias_estimate", soft_bias.item()) 
        self.logger.record("rollout/hard_bias_estimate", (ratio_fairness.max() - ratio_fairness.min()).item()) 
        self.logger.record("rollout/benefit_max", ratio_fairness.max().item()) 
        self.logger.record("rollout/benefit_min", ratio_fairness.min().item()) 

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO_fair",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True, 
    ) -> "PPO_fair":

        return super(PPO_fair, self).learn(            
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,                   
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
    

def smooth_max(x,beta):
    '''
    when beta<0, this is smooth_min
    x: (num_groups,), the R_U/R_B values of each group
    '''
    assert isinstance(x,torch.Tensor), 'x should be a torch.Tensor'
    assert len(x.size()) == 1, 'x should be flat'
    y = x * beta
    y = torch.logsumexp(y,dim=0)
    y = y / beta
    
    # print('y now should be a number: ',y) # xyc test
    # print('min:{}   max:{}'.format(x.min(),x.max())) # xyc test
    return y

def soft_bias_value_and_gradient(x,beta):
    '''
    soft_bias = smooth_max(x,beta) - smooth_max(x,-beta)
    compute the value of soft_bias and the gradient of soft_bias w.r.t. the input x

    If num_group == 2, use hard bias instead
    '''
    assert beta is not None, 'beta for computing the soft bias is None. Please specify it'
    assert isinstance(x, torch.Tensor), 'the type of input should be torch.Tensor'
    num_groups = x.size(0)
    assert num_groups > 1, 'There should be at least two groups in the environment'
    
    if num_groups == 2:
        bias = x.max() - x.min()
        bias_grad = torch.ones(2)
        bias_grad[torch.argmin(x)] = -1
        return bias, bias_grad

    # following: num_groups > 2
    assert num_groups > 2, 'When there are only two groups, do not need to use soft_bias!'

    x.requires_grad = True

    soft_max = smooth_max(x,beta)
    soft_min = smooth_max(x,-beta)

    soft_bias = soft_max - soft_min
    soft_bias_grad = torch.autograd.grad(outputs=soft_bias, inputs=x)[0].detach()
    x.requires_grad = False

    # xyc test:
    # print('by auto grad, the first element of the gradient is ',soft_bias_grad[0] )
    # eps = 1e-3
    # x[0] += eps
    # soft_max = smooth_max(x,beta)
    # soft_min = smooth_max(x,-beta)
    # print('by finite difference, it is ', (soft_max-soft_min -soft_bias)/eps)
    # test ends

    return (soft_bias).detach(), soft_bias_grad