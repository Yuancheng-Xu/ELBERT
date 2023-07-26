import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# for smoothing
from scipy.ndimage.filters import gaussian_filter1d


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def collect_data_csv(exp_path, seeds, timesteps=245):
    results = np.zeros((seeds, timesteps, 3))
    actual_len = 1e10
    count = 0
    for seed in range(seeds):
        try:
            data_pth  = os.path.join("%s_expindex_%d" % (exp_path, seed) ,'eval.csv') 
            data = pd.read_csv(data_pth, sep=',', header=0)
            num_samples = data['num_timesteps'].to_numpy()
            return_arr = data['return'].to_numpy()
            bias = data['bias'].to_numpy()

            if actual_len > min(bias.size, timesteps):
                actual_len = min(bias.size, timesteps)
        
            results[count, :actual_len, 0] = num_samples[:actual_len]
            results[count, :actual_len, 1] = return_arr[:actual_len]
            results[count, :actual_len, 2] = bias[:actual_len]

            count += 1
        except:
            print("%s, exp index %d failed!" % (exp_path, seed))

    results = results[:count,:actual_len,:]

    return results


def bootstrapping(data, num_per_group, num_of_group):
    new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
    return new_data


def generate_confidence_interval(ys, number_per_g = 30, number_of_g = 100, low_percentile = 1, high_percentile = 99):
    means = []
    mins =[]
    maxs = []
    for i,y in enumerate(ys.T):
        y = bootstrapping(y, number_per_g, number_of_g)
        means.append(np.mean(y))
        mins.append(np.percentile(y, low_percentile))
        maxs.append(np.percentile(y, high_percentile))
    return np.array(means), np.array(mins), np.array(maxs)


def plot_ci(plt, x, y, num_runs, num_dots, linestyle='-', linewidth=3, transparency=0.2, c='red', sigma=2.0, label=None):
    assert (x.ndim==1) and (y.ndim==2)
    assert(x.size==num_dots) and (y.shape==(num_runs,num_dots))
    y_mean, y_min, y_max = generate_confidence_interval(y)
    y_mean = gaussian_filter1d(y_mean, sigma=sigma)
    y_max = gaussian_filter1d(y_max, sigma=sigma)
    y_min = gaussian_filter1d(y_min, sigma=sigma)
    plt.plot(x, y_mean, linestyle=linestyle, linewidth=linewidth, color=c, label=label)
    plt.fill_between(x, y_min, y_max, alpha=transparency, color=c)
    return


def load_and_plot_rb(exp_dict, xlim=[0, 5e6],rewards_ylim=None, bias_ylim=None, timesteps=245, sigma_smooth=5.0):
    fig = plt.figure(figsize=(14, 6))

    plt1 = fig.add_subplot(1, 2, 1)
    plt1.set_title("Rewards", fontsize=16)
    plt1.set_ylabel('Rewards', fontsize=16)
    plt1.set_xlabel('Training timestep', fontsize=16)

    plt2 = fig.add_subplot(1, 2, 2)
    plt2.set_title("Bias", fontsize=16)
    plt2.set_ylabel('Bias', fontsize=16)
    plt2.set_xlabel('Training timestep', fontsize=16)

    for model in exp_dict.keys():
        data = collect_data_csv(exp_dict[model]["dir"], exp_dict[model]["seeds"], timesteps)
        indexes = data[:, :, 0]
        if model=="CPO":
            print(model)
            indexes = indexes*7
        rewards = data[:, :, 1]
        bias = data[:, :, 2]
        rewards = gaussian_filter1d(rewards, sigma=sigma_smooth)
        bias = gaussian_filter1d(bias, sigma=sigma_smooth)
        num_runs = indexes.shape[0]
        num_dots = indexes.shape[1]
        plot_ci(plt1, indexes[0], rewards, num_runs=num_runs, num_dots=num_dots, linewidth=2, c=exp_dict[model]["color"], label=exp_dict[model]["label"])
        plot_ci(plt2, indexes[0], np.abs(bias), num_runs=num_runs, num_dots=num_dots, linewidth=2, c=exp_dict[model]["color"], label=exp_dict[model]["label"])

    if rewards_ylim is not None:
        plt1.set_ylim((rewards_ylim[0], rewards_ylim[1]))
    plt1.set_xlim(xlim)
    plt1.grid()
    plt1.legend(loc='lower right')

    if bias_ylim is not None:
        plt2.set_ylim((bias_ylim[0], bias_ylim[1]))
    plt2.set_xlim(xlim)
    plt2.grid()
    
    return fig
