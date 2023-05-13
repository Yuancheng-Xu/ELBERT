import torch
import numpy as np

########## Experiment Setup Parameters ##########
#EXP_DIR = './experiments/advantage_regularized_ppo/'
EXP_DIR = {
    'A-PPO': './experiments/advantage_regularized_ppo',
    'A-PPO_app': './experiments/advantage_regularized_ppo_app',
    'R-PPO': './experiments/reward_regularized_ppo',
    'G-PPO': './experiments/greedy_ppo',
}
SAVE_DIR = f'{EXP_DIR}/models/'

EXP_DIR_AUTO = {
    'A-PPO': './auto_experiments/A-PPO',
    'A-PPO_app': './auto_experiments/advantage_regularized_ppo_app',
    'R-PPO': './auto_experiments/R-PPO',
    'G-PPO': './auto_experiments/G-PPO',
}
SAVE_DIR_AUTO = f'{EXP_DIR_AUTO}/models/'

# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'A-PPO': './experiments/advantage_regularized_ppo/models/rl_model_200000_steps',

}
# CPO model paths to evaluate
CPO_EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'CPO': 'cpo/save-dir/cpo_agent.pt'
}

########## Env Parameters ##########
# ERIC orginal setting
INFECTION_PROBABILITY = 0.5
INFECTED_EXIT_PROBABILITY = 0.005
DISABLE_PROBILITY = 0.0

NUM_TREATMENTS = 1
BURNIN = 1
GRAPH_NAME = 'karate'
EP_TIMESTEPS = 20  # Number of steps in the experiment.

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 10_000_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [512, 512, dict(vf=[512, 256], pi=[512, 256])])
SAVE_FREQ = 20480 # 10000
SAVE_FREQ_AUTO = 10000000000000
EVAL_FREQ_AUTO = 5
REGULARIZE_ADVANTAGE = True  # Regularize advantage?
# Weights for percent healthy and delta term in reward
ZETA_0 = 1
ZETA_1 = 0

# Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper
BETA_0 = 1
BETA_1 = 0.1
BETA_2 = 0.1
# Threshold for delta
OMEGA = 0.05

########## Eval Parameters ##########
# Weights for percent healthy term and delta term in reward
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 0.1

TRANSITION_MATRIX = np.array([
      [0, 0, 0],
      [0, 1 - INFECTED_EXIT_PROBABILITY, INFECTED_EXIT_PROBABILITY],
      [DISABLE_PROBILITY, 0, 1 - DISABLE_PROBILITY]])