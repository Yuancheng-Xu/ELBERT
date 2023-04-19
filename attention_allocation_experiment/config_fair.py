import torch

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/fair_ppo/'

########## Training Parameters ##########
# TRAIN_TIMESTEPS = 10_000_000  # Total train time
# LEARNING_RATE = 0.00001
# POLICY_KWARGS = dict(activation_fn=torch.nn.ReLU,
#                      net_arch = [128, 128, dict(vf=[128, 64], pi=[128, 64])])  # actor-critic architecture
POLICY_KWARGS_fair = dict(activation_fn=torch.nn.ReLU,
                     net_arch = dict(vf=[256, 128], pi=[256, 128])) # new sb3: shared layers is not defined in net_arch here. 
SAVE_FREQ = 1e8  # save frequency in timesteps: don't save model for now
BUFFER_SIZE_TRAINING = 4096 # only for training; for evaluation, the buffer_size = env.ep_timesteps, the number of steps in one episode

BETA_SMOOTH = 5 # for computing soft_bias

########## Evaluation ##########
EVAL_NUM_EPS = 3 # number of episodes for evaluation
EVAL_INTERVAL = 5 # number of training rollout per evaluation (1 training rollout is BUFFER_SIZE_TRAINING samples)
EP_TIMESTEPS_EVAL = 1024 # Number of steps per episode. 
########## Eval Parameters ##########
EVAL_ZETA_0 = 1
EVAL_ZETA_1 = 0.25



########## Env Parameters ##########
N_LOCATIONS = 5
N_ATTENTION_UNITS = 6
EP_TIMESTEPS = 1024
INCIDENT_RATES = [8, 6, 4, 3, 1.5]
DYNAMIC_RATE = 0.1
# Number of timesteps remembered in observation history
OBS_HIST_LEN = 8

# Weights for incidents seen, missed incidents, and delta in reward for the attention allocation environment
ZETA_0 = 1
ZETA_1 = 0.25
ZETA_2 = 0  # 0 means no delta penalty in the reward (should only be non-zero for R-PPO)


