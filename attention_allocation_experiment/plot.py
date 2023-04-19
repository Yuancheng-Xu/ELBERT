import sys
import os
sys.path.insert(1, '/cmlscratch/xic/FairRL/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attention_allocation_experiment.config_fair import EXP_DIR


def plot_return_bias(exp_path):
    
    # read data
    data_pth  = os.path.join(EXP_DIR,exp_path,'eval.csv') 
    data = pd.read_csv(data_pth, sep=',', header=0)
    
    # to numpy
    num_samples = data['num_timesteps'].to_numpy()

    return_arr = data['cash'].to_numpy() # this is the reward averaged across timesteps and episodes, as in attention_allocation_experiment.agents.ppo.sb3.utils_fair
    benefit_max = data['benefit_max'].to_numpy()
    benefit_min = data['benefit_min'].to_numpy()
    bias = data['bias'].to_numpy()
    
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (8,8), sharex=True)
    fig.suptitle('exp path: {}'.format(exp_path))
    # cash
    ax1.plot(num_samples,return_arr)
    ax1.set_ylabel('return')
    ax1.set_title('average return / episode_len')
    # TPR
    ax2.plot(num_samples,benefit_max,label='benefit_max')
    ax2.plot(num_samples,benefit_min,label='benefit_min')
    ax2.legend()
    ax2.set_ylabel('benefit')
    ax2.set_title('benefit_min max and min')
    # bias
    ax3.plot(num_samples,bias)
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Bias')
    ax3.axhline(y=0, color='r', linestyle='-')
    ax3.set_title('Bias')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # fig.show()
    fig.savefig(os.path.join(EXP_DIR,exp_path,'result.png'))

