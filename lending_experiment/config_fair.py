import torch

# from lending_experiment.agents.human_designed_policies.threshold_policies import ThresholdPolicy

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/fair_ppo/' # all experimental results will be under this directory

########## Training Parameters ##########
TRAIN_TIMESTEPS = 2_000_000  # Total train env.steps
LEARNING_RATE = 0.00001
# POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
#                      net_arch = [256, 256, dict(vf=[256, 128], pi=[256, 128])]) # xyc note: this is older sb3 used by Eric, which allowed shared layers
POLICY_KWARGS_fair = dict(activation_fn=torch.nn.ReLU,
                     net_arch = dict(vf=[256, 128], pi=[256, 128])) # new sb3: shared layers is not defined net_arch here. 
SAVE_FREQ = 100000000 # Don't save models.  original: 10000

BUFFER_SIZE_TRAINING = 4096 # only for training; for evaluation, the buffer_size = env.ep_timesteps, the number of steps in one episode

########## Evaluation ##########
EVAL_NUM_EPS = 3 # number of episodes for evaluation
EVAL_INTERVAL = 5 # number of rollout per evaluation (1 rollout is env.ep_timesteps steps during evaluation)
EP_TIMESTEPS_EVAL = 10000 # Number of steps per episode. Eric's paper use EP_TIMESTEPS_EVAL = 10000 for evaluation and EP_TIMESTEPS = 2000 for traning


########## Env Parameters ##########
DELAYED_IMPACT_CLUSTER_PROBS = (
    (0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0),
    (0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0),
)
NUM_GROUPS = 2
GROUP_0_PROB = 0.5
BANK_STARTING_CASH= 10000
INTEREST_RATE = 1
CLUSTER_SHIFT_INCREMENT= 0.01
CLUSTER_PROBABILITIES = DELAYED_IMPACT_CLUSTER_PROBS
EP_TIMESTEPS = 2048 # done=True if rollout for more than ep_timesteps steps by PPO_wrapper; in lending env, done =True also when bank cash < one loan

# Note: in LendingReward_fair, zeta is not used anymore 
# Weights for delta bank cash and delta terms in the reward for the lending environment
# ZETA_0 = 1  
# ZETA_1 = 0  # 0 means no delta penalty in the reward (should only be non-zero for R-PPO)



