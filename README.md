# Principled RL Fairness

# Current status
The current implementation differs from the previous one as follows

Codebase:
* All PPO methods (GPPO, APPO, RPPO, ours) use the same codebase (including all sb3 files, env wrapper, ActorCriticPolicy, architecture, observation space, buffer_size, etc). Baselines will be as slow as our method. 
* sb3 files are shared across all envs. To extract environment specific information such as "incident rate", go to the common sb3 file and save those information there (using if 'attention' in str(env))
* The script file (except lending env) is more automatic. 

Our method implementation
* Now, the chain rule is first applied to advantages (including main and fairness advantages), then use the clipped loss of PPO. As comparison, in the old implementation, we first apply clipped loss to compute each gradient, and then compute the chain rule.


# Original code
Authors: Eric Yang Yu, Zhizhen Qin, Min Kyung Lee, Sicun Gao \
NeurIPS (Conference on Neural Information Processing Systems) 2022

This is the code implementation for the NeurIPS 2022 [paper](https://arxiv.org/abs/2210.12546) above. 
Code and environments are adapted from the original Google ML-fairness-gym [repo](https://github.com/google/ml-fairness-gym).
