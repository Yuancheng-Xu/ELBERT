import sys
import os
sys.path.insert(1, '/cmlscratch/xic/FairRL/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# for smoothing
from scipy.ndimage.filters import gaussian_filter1d

from lending_experiment.config_fair import EXP_DIR


def plot_cash_bias(exp_path, save=True, smooth=-1):
    
    # read data
    data_pth  = os.path.join(EXP_DIR,exp_path,'eval.csv') 
    data = pd.read_csv(data_pth, sep=',', header=0)
    
    # to numpy
    num_samples = data['num_timesteps'].to_numpy()

    cash = data['cash'].to_numpy()
    tpr_0 = data['tpr_0'].to_numpy()
    tpr_1 = data['tpr_1'].to_numpy()
    bias = data['bias'].to_numpy()
#     bias = np.absolute(bias)

    if smooth > 0:
        cash = gaussian_filter1d(cash, sigma=smooth)
        tpr_0 = gaussian_filter1d(tpr_0, sigma=smooth)
        tpr_1 = gaussian_filter1d(tpr_1, sigma=smooth)
        bias = gaussian_filter1d(bias, sigma=smooth)
    
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (8,8), sharex=True)
    fig.suptitle('exp path: {}'.format(exp_path))
    # cash
    ax1.plot(num_samples,cash)
    ax1.set_ylabel('Cash')
    ax1.set_title('Cash')
    ax1.grid()
    # TPR
    ax2.plot(num_samples,tpr_0,label='Group 0')
    ax2.plot(num_samples,tpr_1,label='Group 1')
    ax2.legend()
    ax2.set_ylabel('TPR')
    ax2.set_title('TPR')
    ax2.grid()
    # bias
    ax3.plot(num_samples,bias)
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Bias')
    ax3.axhline(y=0, color='r', linestyle='-')
    ax3.set_title('Bias = TPR_0 - TPR_1')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax3.grid()
    if save:
        fig.savefig(os.path.join(EXP_DIR,exp_path,'result.png'))
    else:
        fig.show()

# +
# bias_list = [0,20,50,100,200,400,1000]
# exp_path_base = 'Chenghao_env_1/lr_1e-6_samples_5e6/b_'
# for b in bias_list:
#     exp_path = exp_path_base + str(b)
#     plot_cash_bias(exp_path,save=False,smooth=2)
