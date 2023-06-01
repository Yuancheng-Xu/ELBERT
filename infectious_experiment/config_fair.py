import torch

########## Experiment Setup Parameters ##########
EXP_DIR = './experiments/fair_ppo/' # all experimental results will be under this directory

########## Training Parameters ##########
POLICY_KWARGS_fair = dict(activation_fn=torch.nn.ReLU,
                     net_arch = dict(vf=[512, 256], pi=[512, 256])) # new sb3: shared layers is not defined in net_arch here. 
SAVE_FREQ = 100000000 # Don't save models.  original: 10000
BUFFER_SIZE_TRAINING = 5000 # xyc used 60 # only for training; for evaluation, the buffer_size = env.ep_timesteps, the number of steps in one episode

########## Evaluation ##########
EVAL_NUM_EPS = 100 # number of episodes for evaluation; Eric: 200
EVAL_INTERVAL = 5 # xyc used 200 # number of training rollout per evaluation (1 training rollout is BUFFER_SIZE_TRAINING samples)
EP_TIMESTEPS_EVAL = 20

# note that in the wrapper_fair, I use the original Eric's reward_fn 
# (just for training; evaluation still uses InfectiousReward_fair)
ZETA_0 = 1
ZETA_1 = 0 # RPPO: 0.1; GPPO and ours: 0

########## Env Parameters ##########
INFECTION_PROBABILITY = 0.5
INFECTED_EXIT_PROBABILITY = 0.005 # Pr(I->R)
NUM_TREATMENTS = 1
BURNIN = 1
GRAPH_NAME = 'karate'
EP_TIMESTEPS = 20  # Number of steps in the experiment.

'''
xyc:
BUFFER_SIZE_TRAINING needs to be a multiply of EP_TIMESTEPS_EVAL right now (a buffer is a multiply of episode)
Otherwise there will be errors
'''


