import sys
import os
sys.path.insert(1, '/cmlscratch/xic/FairRL/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# for smoothing
from scipy.ndimage.filters import gaussian_filter1d

from infectious_experiment.config_fair import EXP_DIR


def plot_return_bias(exp_path, save=True, smooth=-1):
    
    # read data
    data_pth  = os.path.join(EXP_DIR,exp_path,'eval.csv') 
    data = pd.read_csv(data_pth, sep=',', header=0)
    
    # to numpy
    num_samples = data['num_timesteps'].to_numpy()

    return_arr = data['return'].to_numpy() # this is the reward averaged across timesteps and episodes, as in infectious_experiment.agents.ppo.sb3.utils_fair
    benefit_max = data['benefit_max'].to_numpy()
    benefit_min = data['benefit_min'].to_numpy()
    bias = data['bias'].to_numpy()
    
    if smooth > 0:
        return_arr = gaussian_filter1d(return_arr, sigma=smooth)
        benefit_max = gaussian_filter1d(benefit_max, sigma=smooth)
        benefit_min = gaussian_filter1d(benefit_min, sigma=smooth)
        bias = gaussian_filter1d(bias, sigma=smooth)
    
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (8,8), sharex=True)
    fig.suptitle('exp path: {}'.format(exp_path))
    # cash
    ax1.plot(num_samples,return_arr)
    ax1.set_ylabel('return')
    ax1.set_title('average return / episode_len')
    ax1.grid()
    # TPR
    ax2.plot(num_samples,benefit_max,label='benefit_max')
    ax2.plot(num_samples,benefit_min,label='benefit_min')
    ax2.legend()
    ax2.set_ylabel('benefit')
    ax2.set_title('benefit max and min')
    # bias
    ax3.plot(num_samples,bias)
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Bias')
    ax3.axhline(y=0, color='r', linestyle='-')
    ax3.set_title('Bias')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax3.grid()
    if save:
        fig.savefig(os.path.join(EXP_DIR,exp_path,'result.png'))
    else:
        fig.show()

# +
# bias_list = [0,0.1,0.5,1,2,4,5,10,15,20,25,30,40,50,100,200]
# exp_path_base = 'lr_1e-6_samples_5e6/b_'
# for b in bias_list:
#     exp_path = exp_path_base + str(b)
#     plot_return_bias(exp_path,save=False)

# +
# bias_list = [0,1,5,10,20,30,40,50]
# bias_list = [1,10,20,30]
# exp_path_base = 'Chenghao_p_0.2_buffer5000/lr_1e-5_samples_5e6/b_'
# for b in bias_list:
#     exp_path = exp_path_base + str(b)
#     exp_path = exp_path_base + str(b) + '_s_0'
#     plot_return_bias(exp_path,save=False,smooth=2)

# +
# b = 0.5
# plot_return_bias(exp_path_base + str(b) + '_s_0',save=False,smooth=2)
# plot_return_bias(exp_path_base + str(b) + '_s_1',save=False,smooth=2)

# +
# b = 1
# plot_return_bias(exp_path_base + str(b) + '_s_0',save=False,smooth=2)
# plot_return_bias(exp_path_base + str(b),save=False,smooth=2)
