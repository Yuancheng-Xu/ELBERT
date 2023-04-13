import sys
import os
sys.path.insert(1, '/cmlscratch/xic/FairRL/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lending_experiment.config_fair import EXP_DIR


def plot_cash_bias(exp_path):
    
    # read data
    data_pth  = os.path.join(EXP_DIR,exp_path,'eval.csv') 
    data = pd.read_csv(data_pth, sep=',', header=0)
    
    # to numpy
    num_samples = data['num_timesteps'].to_numpy()

    cash = data['cash'].to_numpy()
    tpr_0 = data['tpr_0'].to_numpy()
    tpr_1 = data['tpr_1'].to_numpy()
    bias = data['bias'].to_numpy()
    
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (8,8), sharex=True)
    fig.suptitle('exp path: {}'.format(exp_path))
    # cash
    ax1.plot(num_samples,cash)
    ax1.set_ylabel('Cash')
    ax1.set_title('Cash')
    # TPR
    ax2.plot(num_samples,tpr_0,label='Group 0')
    ax2.plot(num_samples,tpr_1,label='Group 1')
    ax2.legend()
    ax2.set_ylabel('TPR')
    ax2.set_title('TPR')
    # bias
    ax3.plot(num_samples,bias)
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Bias')
    ax3.axhline(y=0, color='r', linestyle='-')
    ax3.set_title('Bias = TPR_0 - TPR_1')
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    # fig.show()
    fig.savefig(os.path.join(EXP_DIR,exp_path,'result.png'))

# +
# # read data
# exp_path = 'b_50'
# data_pth  = os.path.join(EXP_DIR,exp_path,'eval.csv') 
# data = pd.read_csv(data_pth, sep=',', header=0)

# # to numpy
# num_samples = data['num_timesteps'].to_numpy()

# cash = data['cash'].to_numpy()
# tpr_0 = data['tpr_0'].to_numpy()
# tpr_1 = data['tpr_1'].to_numpy()
# bias = data['bias'].to_numpy()

# +
# # plot
# fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (8,8), sharex=True)
# fig.suptitle('exp path: {}'.format(exp_path))
# # cash
# ax1.plot(num_samples,cash)
# ax1.set_ylabel('Cash')
# ax1.set_title('Cash')
# # TPR
# ax2.plot(num_samples,tpr_0,label='Group 0')
# ax2.plot(num_samples,tpr_1,label='Group 1')
# ax2.legend()
# ax2.set_ylabel('TPR')
# ax2.set_title('TPR')
# # bias
# ax3.plot(num_samples,bias)
# ax3.set_xlabel('Samples')
# ax3.set_ylabel('Bias')
# ax3.axhline(y=0, color='r', linestyle='-')
# ax3.set_title('Bias = TPR_0 - TPR_1')
# ax3.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# fig.show()

# +
# plt.plot(num_samples,cash)
# plt.xlabel('Samples')
# plt.ylabel('Cash')
# plt.title('Cash')
# plt.show()

# +
# plt.plot(num_samples,tpr_0,label='Group 0')
# plt.plot(num_samples,tpr_1,label='Group 1')
# plt.legend()
# plt.xlabel('Samples')
# plt.ylabel('TPR')
# plt.title('TPR')
# plt.show()

# +
# plt.plot(num_samples,bias)
# plt.xlabel('Samples')
# plt.ylabel('Bias')
# plt.axhline(y=0, color='r', linestyle='-')
# plt.title('Bias = TPR_0 - TPR_1')
# plt.show()
