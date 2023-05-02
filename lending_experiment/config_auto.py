import torch

from lending_experiment.agents.human_designed_policies.threshold_policies import ThresholdPolicy

########## Experiment Setup Parameters ##########
# EXP_DIR = './experiments/advantage_regularized_ppo/'
EXP_DIR = {
    'A-PPO': './experiments/advantage_regularized_ppo',
    'A-PPO_app': './experiments/advantage_regularized_ppo_app',
    'R-PPO': './experiments/reward_regularized_ppo',
    'G-PPO': './experiments/greedy_ppo',
}

EXP_DIR_AUTO = {
    'A-PPO': './auto_experiments/A-PPO',
    'R-PPO': './auto_experiments/R-PPO',
    'G-PPO': './auto_experiments/G-PPO',
}

SAVE_DIR = f'{EXP_DIR}/models/'
SAVE_DIR_AUTO = f'{EXP_DIR_AUTO}/models/'
# PPO model paths to evaluate
EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'A-PPO': './experiments/advantage_regularized_ppo/models/rl_model_100000_steps',

}
# CPO model paths to evaluate
CPO_EVAL_MODEL_PATHS = {
    # Sample model evaluation path
    'CPO': 'cpo/save-dir/cpo_agent.pt'
}

########## Env Parameters ##########
DELAYED_IMPACT_CLUSTER_PROBS_0 = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)

DELAYED_IMPACT_CLUSTER_PROBS_1 = (
    (0.05, 0.05, 0.0, 0.2, 0.1, 0.3, 0.3),
    (0.0, 0.0, 0.3, 0.3, 0.3, 0.05, 0.05),
)

DELAYED_IMPACT_CLUSTER_PROBS_2 = (
    (0.2, 0.05, 0.2, 0.0, 0.0, 0.5, 0.05),
    (0.0, 0.3, 0.4, 0.2, 0.1, 0.0, 0.0),
)

DELAYED_IMPACT_CLUSTER_PROBS = {
    0:DELAYED_IMPACT_CLUSTER_PROBS_0,
    1:DELAYED_IMPACT_CLUSTER_PROBS_1,
    2:DELAYED_IMPACT_CLUSTER_PROBS_2
}

ALTER_RATE = [0.3, 0.1]
# TRANSITION_DYNAMICS = [
#     [
#         [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
#         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#     ],
#     [
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
#     ],
# ]

TRANSITION_DYNAMICS = [
    [
        [0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.7, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.7, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.7, 0.1, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.2, 0.2, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.8],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ],
    [
        [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.7, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.7, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.7, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.7, 0.1],
    ],
]

NUM_GROUPS = 2
NUM_FEATURES = 7
GROUP_0_PROB = 0.5
BANK_STARTING_CASH= 10000
INTEREST_RATE = 1
CLUSTER_SHIFT_INCREMENT= 0.01
CLUSTER_PROBABILITIES = DELAYED_IMPACT_CLUSTER_PROBS_0
EP_TIMESTEPS = 2000
MAXIMIZE_REWARD = ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = ThresholdPolicy.EQUALIZE_OPPORTUNITY

########## PPO Train Parameters ##########
TRAIN_TIMESTEPS = 10_000_000  # Total train time
LEARNING_RATE = 0.00001
POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
                     net_arch = [256, 256, dict(vf=[256, 128], pi=[256, 128])])
SAVE_FREQ = 10000
SAVE_FREQ_AUTO = 100000000000  # save frequency in timesteps
EVAL_FREQ_AUTO = 5
REGULARIZE_ADVANTAGE = True
# Weights for delta bank cash and delta terms in the reward for the lending environment
ZETA_0 = 1
ZETA_1 = 0  # 0 means no delta penalty in the reward (should only be non-zero for R-PPO)

# Weights for advantage, value-thresholding, and decrease-in-violation terms in Eq. 3 of the paper
BETA_0 = 1
BETA_1 = 0.25
BETA_2 = 0.25

# Threshold for delta
OMEGA = 0.005

########## Eval Parameters ##########
# Weights for delta bank cash and delta terms in the reward for the lending environment
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 2
BURNIN = 0  # Number of steps before applying the threshold policy.

