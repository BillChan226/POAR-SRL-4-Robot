import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import json
from ipdb import set_trace as tt
from rl_baselines.visualize import  movingAverage, loadCsv,loadData
from replay.aggregate_plots import lightcolors, darkcolors, millions
from srl_zoo.utils import printGreen, printRed, printYellow
from replay.pipeline import GatherExperiments, plotGatheredData, loadEpisodesData


def comparePlots(path, algo, y_limits, title="Learning Curve",
                 timesteps=False, truncate_x=-1, no_display=False,normalization=False):
    """
    :param path: (str) path to the folder where the plots are stored
    :param plots: ([str]) List of saved plots as npz file
    :param y_limits: ([float]) y-limits for the plot
    :param title: (str) plot title
    :param timesteps: (bool) Plot timesteps instead of episodes
    :param truncate_x: (int) Truncate the experiments after n ticks on the x-axis
    :param no_display: (bool) Set to true, the plot won't be displayed (useful when only saving plot)
    """
    # env = 'OmnirobotEnv-v0'
    # env = 'MobileRobotGymEnv-v0'
    folders = []
    legends=[]

    # path = os.path.join(path, os.listdir(path)[0])

    for folder in os.listdir(path):
        folders_list = []
        tmp_path = os.path.join(path, folder)
        tmp_path = os.path.join(tmp_path, os.listdir(tmp_path)[0])
        legends.append(folder)
        for f in os.listdir(tmp_path):
            folders_list.append(os.path.join(tmp_path, f))
        folders.append(folders_list)


    x_list,y_list=[],[]
    for folder in folders:
        printGreen("Folder name {}".format(folder))

        x, y =GatherExperiments(folder, algo,  window=40, title=title, min_num_x=-1, timesteps=timesteps, output_file="")

        print(len(x))
        x_list.append(x)
        y_list.append(y)
    printGreen(np.array(x_list).shape)
    # printGreen('y_list shape {}'.format(np.array(y_list[1]).shape))
    plotGatheredData(x_list,y_list,y_limits,timesteps,title,legends,no_display,truncate_x,normalization)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trained agent")
    parser.add_argument('-i', '--input-dir', help='folder with the plots as npz files', type=str, required=True)
    parser.add_argument('-t', '--title', help='Plot title', type=str, default='Learning Curve')
    parser.add_argument('--timesteps', help='Plot title', action='store_true', default=False)
    args = parser.parse_args()
    comparePlots(path=args.input_dir, algo='ppo2', y_limits=[0,200], timesteps=args.timesteps)
