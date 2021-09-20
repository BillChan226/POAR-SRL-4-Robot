"""
baseline benchmark script for openAI RL Baselines
"""
import os
import argparse
import subprocess

import yaml
import numpy as np

from rl_baselines.registry import registered_rl
from environments.registry import registered_env
from state_representation.registry import registered_srl
from state_representation import SRLType
from srl_zoo.utils import printGreen, printRed, printYellow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


def main():
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines Benchmark",
                                     epilog='After the arguments are parsed, the rest are assumed to be arguments for' +
                                            ' rl_baselines.train')
    parser.add_argument('--algo', type=str, default='poar', help='OpenAI baseline to use',
                        choices=list(registered_rl.keys()))
    parser.add_argument('--env', type=str, nargs='+', default=["OmnirobotEnv-v0","MobileRobotGymEnv-v0"], help='environment ID(s)',
                        choices=list(registered_env.keys()))
    parser.add_argument('--timesteps', type=int, default=2e6, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--num-iteration', type=int, default=15,
                        help='number of time each algorithm should be run for each unique combination of environment ' +
                             ' and srl-model.')
    parser.add_argument('--seed', type=int, default=0,
                        help='initial seed for each unique combination of environment and srl-model.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-cpu', type=str, default='2')
    # returns the parsed arguments, and the rest are assumed to be arguments for rl_baselines.train
    args, train_args = parser.parse_known_args()
    envs = ["OmnirobotEnv-v0"]
    envs = ["KukaRandButtonGymEnv-v0"]
    seeds = np.arange(args.num_iteration)

    printGreen("\nRunning {} benchmarks {} times...".format(args.algo, args.num_iteration))
    print("environments:\t{}".format(envs))
    print("verbose:\t{}".format(args.verbose))
    print("timesteps:\t{}".format(args.timesteps))


    # 'reconstruction, forward, inverse, state_entropy, reward'
    srl_weights = [
                    # compare the loss on the autoencoder
                    #["autoencoder:1:200", "reward:5:-1", "inverse:2:4", "forward:0:-1", "entropy:0:-1"],  # split
                    ["autoencoder:1:120", "reward:5:-1", "inverse:2:50", "forward:1:50", "entropy:0:-1"],  # BEST !! split
                    ["autoencoder:1:220", "reward:5:-1", "inverse:2:-1", "forward:1:-1", "entropy:0:-1"],
                    #["autoencoder:1:100", "reward:5:-1", "inverse:2:50", "forward:2:-1", "entropy:0:-1"],  # split
                    # ["autoencoder:1:100", "reward:1:-1", "inverse:5:4", "forward:1:4", "entropy:0:-1"],  # split
                   ]
    srl_name = ['a','r','i','f','e']
    if args.verbose:
        # None here means stdout of terminal for subprocess.call
        stdout = None
    else:
        # shut the g** d** mouth
        stdout = open(os.devnull, 'w')
    tasks = ['-cc']#, '-sc', '-esc']

    for i in range(1, args.num_iteration+1):
        for env in envs:
            for task in tasks:
                printGreen(
                    "\nIteration_num={} (seed: {}), Environment='{}', Algo='{}'".format(i, seeds[i-1], env, args.algo))
                for weights in srl_weights:
                    name = "combi"
                    dim = ""
                    for w in weights:
                        if "autoencoder" in w:
                            dim = w.split(":")[-1]
                        if "autoencoder" not in w and int(w.split(":")[-1]) > 0:
                            name = "split"
                            break


                    log_dir = 'logs/POAR-kuka/srl_{}_{}_'.format( name, dim)

                    weight_args = ['--losses']
                    for j, w in enumerate(weights):
                        weight = int(w.split(":")[1])
                        if  weight > 0:
                            log_dir += srl_name[j]+str(weight)
                        weight_args += [str(w)]
                    log_dir += '/'
                    loop_args = ['--seed', str(seeds[i-1]), '--algo', args.algo, '--env', env, '--srl-model', 'raw_pixels',
                                 '--num-timesteps', str(int(args.timesteps)),
                                 '--log-dir', log_dir, '--gpu', str(args.gpu), '--num-cpu', args.num_cpu,
                                 '-r']
                    loop_args += weight_args
                    poar_args = ['--structure', 'srl_autoencoder']
                    loop_args += poar_args
                    printYellow(loop_args)
                    subprocess.call(['python', '-m', 'rl_baselines.train'] + train_args + loop_args, stdout=stdout)


if __name__ == '__main__':
    main()
