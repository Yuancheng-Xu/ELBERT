import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def collect_data(exp_name, seeds, timesteps=245, freq=20480, eval_path='auto_experiments', save_path='results'):
    results = np.zeros((seeds, timesteps, 3))
    for timestep in range(timesteps):
        index = freq*timestep
        for seed in range(seeds):
            exp_path = "%s_%d" % (exp_name, seed)
            result_path = os.path.join(eval_path, exp_path, "eval", "train_%d" % (index))
            rewards = np.load(os.path.join(result_path, "rewards.npy"))
            bias = np.load(os.path.join(result_path, "bias.npy"))
            results[seed, timestep, 0] = index
            results[seed, timestep, 1] = rewards.mean()
            results[seed, timestep, 2] = bias

    np.save(f'{save_path}/{exp_name}.npy', results)
    return 0

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--seeds', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=245)
    parser.add_argument('--freq', type=int, default=20480)
    parser.add_argument('--eval_path', type=str, default='auto_experiments')
    parser.add_argument('--save_path', type=str, default='results')

    args = parser.parse_args()

    collect_data(args.exp_name, args.seeds, args.timesteps, args.freq, args.eval_path, args.save_path)

if __name__ == '__main__':
    main()