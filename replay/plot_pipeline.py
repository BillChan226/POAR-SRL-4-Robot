import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import json

from ipdb import set_trace as tt
from rl_baselines.visualize import movingAverage, loadCsv, loadData
from replay.aggregate_plots import lightcolors, darkcolors, Y_LIM_SHAPED_REWARD, Y_LIM_SPARSE_REWARD, millions
from srl_zoo.utils import printGreen, printRed, printYellow

# Init seaborn
sns.set()
# Style for the title
fontstyle = {'fontname': 'DejaVu Sans', 'fontsize': 16}


def loadEpisodesData(folder):
    """
    :param folder: (str)
    :return: (numpy array, numpy array) or (None, None)
    """
    result, _ = loadCsv(folder)

    if len(result) == 0:
        return None, None

    y = np.array(result)[:, 1]
    x = np.arange(len(y))
    return x, y


def plotGatheredData(x_list, y_list, y_limits, timesteps, title, legends, no_display, truncate_x=-1, normalization=False, figpath=None, exp_name_dict=None):
    assert len(legends) == len(y_list)
    printGreen("{} Experiments".format(len(y_list)))

    lengths = list(map(len, x_list))
    min_x, max_x = np.min(lengths), np.max(lengths)
    if truncate_x > 0:
        min_x = min(truncate_x, min_x)
    x = np.array(x_list[0][:min_x])
    # To reformulize the data by the min_x
    for i in range(len(y_list)):
        y_list[i] = y_list[i][:, :min_x]
    y_list = np.array(y_list)

    #print("Min, Max rewards:", np.min(y_list), np.max(y_list))

    # Normalize the data between 0 and 1.
    if (normalization):
        y_limits = [-0.05, 1.05]
        y_list = (y_list-np.min(y_list))/(np.max(y_list)-np.min(y_list))

    colormap = plt.cm.tab20.colors
    registered_indexes = [0, 4, 6, 5]
    registered_color = {#'GT': colormap[4],        # green
                        'ppo2': colormap[4],          # blue
                        'srl_a5f1i1': colormap[6],  # red
                        # 'ground_truth': colormap[5]
                        }
    all_name = [
        'srl_a5f1i2r5',
        'srl_a1f1i10',
        'srl_a1f10i1',
        'ae',
        'ppo2',
        'aeBN',
        'srl_split_a5i1f1',
        'srl_a1f1i5',
        'srl_a1f1i1',
        'srl_a1f5i1',
        'srl_',
        'aeMlp',
        'srl_a5f1i2r5_compare',
        'srl_a10f1i1',
        'GT',
        'ground_truth',
        'srl_a5f1i1',
        'srl_split_a2r5f1i2'
    ]
    new_colormap = tuple([colormap[k%len(colormap)] for k in range(len(y_list))])

    for k in exp_name_dict:
        if exp_name_dict[k] == 'ppo2':
            ppo2_index = k
        if exp_name_dict[k] == 'a2c':
            a
    ppo2 = y_list[ppo2_index][:, :min_x]
    ppo2_m = np.mean(ppo2, axis=0)
    ppo2_s = np.squeeze(np.asarray(np.std(ppo2, axis=0)))
    ppo2_n = ppo2.shape[0]
    if not os.path.exists('SRL_plot'):
        os.mkdir("SRL_plot")
    #fig = plt.figure(title, figsize=(20, 10))
    for i in range(len(y_list)):
        exp_name = exp_name_dict[i]
        plt.close("all")
        fig = plt.figure(exp_name, figsize=(15, 10))
        label = legends[i]
        y = y_list[i][:, :min_x]
        print('{}: {} experiments'.format(label, len(y)))
        # Compute mean for different seeds
        m = np.mean(y, axis=0)
        # Compute standard error
        s = np.squeeze(np.asarray(np.std(y, axis=0)))
        n = y.shape[0]

        if exp_name == 'ppo2':
            continue

        color = registered_color.get(exp_name, new_colormap[i]) # get color if exp_name is registered, otherwise, new color

        plt.fill_between(x, ppo2_m- ppo2_s / np.sqrt(ppo2_n), ppo2_m + ppo2_s / np.sqrt(ppo2_n), color=colormap[4], alpha=0.3)
        plt.plot(x, ppo2_m, color=colormap[4], label='ppo2', linewidth=2)
        plt.fill_between(x, m - s / np.sqrt(n), m + s / np.sqrt(n), color=colormap[6], alpha=0.3)
        plt.plot(x, m, color=colormap[6], label=label, linewidth=2)
        if timesteps:
            formatter = FuncFormatter(millions)
            plt.xlabel('Number of Timesteps')
            fig.axes[0].xaxis.set_major_formatter(formatter)
        else:
            plt.xlabel('Number of Episodes')
        if (normalization):
            plt.ylabel('Normalized Rewards')
        else:
            plt.ylabel('Rewards')
        plt.title(exp_name, **fontstyle)
        plt.ylim(y_limits)

        plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=16)

        plt.savefig("SRL_plot/{}.png".format(exp_name))



    # for i in range(len(y_list)):
    #     label = legends[i]
    #     y = y_list[i][:, :min_x]
    #
    #     print('{}: {} experiments'.format(label, len(y)))
    #     # Compute mean for different seeds
    #     m = np.mean(y, axis=0)
    #     # Compute standard error
    #     s = np.squeeze(np.asarray(np.std(y, axis=0)))
    #     n = y.shape[0]
    #     exp_name = exp_name_dict[i]
    #     color = registered_color.get(exp_name, new_colormap[i]) # get color if exp_name is registered, otherwise, new color
    #     if exp_name in registered_color:
    #         plt.fill_between(x, m - s / np.sqrt(n), m + s / np.sqrt(n), color=color, alpha=0.3)
    #         plt.plot(x, m, color=color, label=label, linewidth=2)
    

    if timesteps:
        formatter = FuncFormatter(millions)
        plt.xlabel('Number of Timesteps')
        fig.axes[0].xaxis.set_major_formatter(formatter)
    else:
        plt.xlabel('Number of Episodes')
    if(normalization):
        plt.ylabel('Normalized Rewards')
    else:
        plt.ylabel('Rewards')
    plt.title(title, **fontstyle)
    plt.ylim(y_limits)

    plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=16)
    if figpath is not None:
        plt.savefig(figpath)
    if not no_display:
        plt.show()


def GatherExperiments(folders, algo,  window=40, title="", min_num_x=-1,
                      timesteps=False, output_file="",):
    """
    Compute mean and standard error for several experiments and plot the learning curve
    :param folders: ([str]) Log folders, where the monitor.csv are stored
    :param window: (int) Smoothing window
    :param algo: (str) name of the RL algo
    :param title: (str) plot title
    :param min_num_x: (int) Minimum number of episode/timesteps to keep an experiment (default: -1, no minimum)
    :param timesteps: (bool) Plot timesteps instead of episodes
    :param y_limits: ([float]) y-limits for the plot
    :param output_file: (str) Path to a file where the plot data will be saved
    :param no_display: (bool) Set to true, the plot won't be displayed (useful when only saving plot)
    """
    y_list = []
    x_list = []
    ok = False
    for folder in folders:
        if timesteps:
            x, y = loadData(folder, smooth=1, bin_size=100)
            if x is not None:
                x, y = np.array(x), np.array(y)
        else:
            x, y = loadEpisodesData(folder)

        if x is None or (min_num_x > 0 and y.shape[0] < min_num_x):
            printRed("Skipping {}".format(folder))
            continue

        if y.shape[0] <= window:
            printYellow("Folder {}".format(folder))
            printRed("Not enough episodes for current window size = {}".format(window))
            continue
        ok = True
        y = movingAverage(y, window)
        y_list.append(y)
        print(len(x))
        # Truncate x
        x = x[len(x) - len(y):]
        x_list.append(x)
    
    if not ok:
        printRed("Not enough data to plot anything with current config." +
                 " Consider decreasing --min-x")
        return

    lengths = list(map(len, x_list))
    min_x, max_x = np.min(lengths), np.max(lengths)

    print("Min x: {}".format(min_x))
    print("Max x: {}".format(max_x))

    for i in range(len(x_list)):
        x_list[i] = x_list[i][:min_x]
        y_list[i] = y_list[i][:min_x]

    x = np.array(x_list)[0]
    y = np.array(y_list)
    # if output_file != "":
    #     printGreen("Saving aggregated data to {}.npz".format(output_file))
    #     np.savez(output_file, x=x, y=y)
    return x, y


def comparePlots(path,  algo, y_limits, title="Learning Curve",
                 timesteps=False, truncate_x=-1, no_display=False, normalization=False, figpath=None,
                 exclude_list=None):
    """
    :param path: (str) path to the folder where the plots are stored
    :param plots: ([str]) List of saved plots as npz file
    :param y_limits: ([float]) y-limits for the plot
    :param title: (str) plot title
    :param timesteps: (bool) Plot timesteps instead of episodes
    :param truncate_x: (int) Truncate the experiments after n ticks on the x-axis
    :param no_display: (bool) Set to true, the plot won't be displayed (useful when only saving plot)
    """
    if exclude_list is None:
        exclude_list = []
    folders = []
    other = []
    legends = []

    env = 'OmnirobotEnv-v0'
    #env = 'MobileRobotGymEnv-v0'
    srl = 'raw_pixels'


    for folder in os.listdir(path):
        folders_srl = []
        other_srl = []
        tmp_path = "{}/{}/".format(path, folder)
        tmp_path = os.path.join(tmp_path, os.listdir(tmp_path)[0])
        tmp_path = os.path.join(tmp_path, os.listdir(tmp_path)[0])
        tmp_path = os.path.join(tmp_path, os.listdir(tmp_path)[0])

        if os.path.exists(tmp_path) and (folder not in exclude_list):  # folder contains algo (e.g. ppo2) subfolder and not in excluded list
            printRed(folder)
            legends.append(folder)
            for f in os.listdir(tmp_path):
                paths = "{}/{}/".format(tmp_path, f)
                env_globals = json.load(open(paths + "env_globals.json", 'r'))
                train_args = json.load(open(paths + "args.json", 'r'))
                if train_args["shape_reward"] == args.shape_reward:
                    folders_srl.append(paths)
                else:
                    other_srl.append(paths)
            folders.append(folders_srl)
            other.append(other_srl)
        else:
            continue


    x_list, y_list = [], []
    exp_name_dict = {}
    for ind, folders_srl in enumerate(folders):
        printGreen("Folder name {}".format(folders_srl))
        x, y = GatherExperiments(folders_srl, algo,  window=40, title=title, min_num_x=-1,
                                 timesteps=timesteps, output_file="")
        print(len(x))
        x_list.append(x)
        y_list.append(y)
        ## HACK: the line below is ugly and not robust code !! TODO
        exp_name_dict[ind] = folders_srl[0].split("/")[3]
    printGreen(np.array(x_list).shape)
    # printGreen('y_list shape {}'.format(np.array(y_list[1]).shape))
    plotGatheredData(x_list, y_list, y_limits, timesteps, title, legends,
                     no_display, truncate_x, normalization, figpath=figpath, exp_name_dict=exp_name_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trained agent")
    parser.add_argument('-i', '--input-dir', help='folder with the plots as npz files', type=str, required=True)
    parser.add_argument('-t', '--title', help='Plot title', type=str, default='Learning Curve')
    parser.add_argument('--episode_window', type=int, default=40,
                        help='Episode window for moving average plot (default: 40)')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Change the y_limit to correspond shaped reward bounds')
    parser.add_argument('--y-lim', nargs=2, type=float, default=[-1, -1], help="limits for the y axis")
    parser.add_argument('--truncate-x', type=int, default=-1,
                        help="Truncate the experiments after n ticks on the x-axis (default: -1, no truncation)")
    parser.add_argument('--timesteps', action='store_true', default=False,
                        help='Plot timesteps instead of episodes')
    parser.add_argument('--no-display', action='store_true', default=False, help='Do not display plot')
    parser.add_argument('--algo', type=str, default='ppo2', help='The RL algorithms result to show')
    parser.add_argument('--norm', action='store_true', default=False,
                        help='To normalize the output by the maximum reward')
    parser.add_argument('--figpath', type=str, default=None, help='Save figure to path.')
    parser.add_argument('--exclude', nargs='+', type=str, default=None, help='SRL models to be excluded.')
    #
    # parser.add_argument('--tasks', type=str, nargs='+', default=["cc"],
    #                     help='The tasks for the robot',
    #                     choices=["cc", "ec", "sqc", "sc"])

    args = parser.parse_args()
    
    y_limits = args.y_lim
    if y_limits[0] == y_limits[1]:
        if args.shape_reward:
            y_limits = Y_LIM_SHAPED_REWARD
        else:
            y_limits = Y_LIM_SPARSE_REWARD
        print("Using default limits:", y_limits)

    ALGO_NAME = args.algo

    x_list = []
    y_list = []

    comparePlots(args.input_dir, args.algo, title=args.title, y_limits=y_limits, no_display=args.no_display,
                 timesteps=args.timesteps, truncate_x=args.truncate_x, normalization=args.norm, figpath=args.figpath,
                 exclude_list=args.exclude)


# python -m replay.plot_pipeline -i logs/OmnirobotEnv-v0 --algo ppo2 --title cc --timesteps
