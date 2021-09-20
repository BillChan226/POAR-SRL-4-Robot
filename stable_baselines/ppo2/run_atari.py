from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from rl_baselines.utils import computeMeanReward
import os
import datetime
from visdom import Visdom
from rl_baselines.visualize import timestepsPlot, episodePlot

LOG_DIR = ""
best_reward = -10000
ALGO_NAME = "PPO"
PLOT_TITLE = "ATARI"
win, win_smooth, win_episodes = None, None, None
N_EPISODES_EVAL = 100
EPISODE_WINDOW = 40


def callback(_locals, _globals):
    global best_reward, ALGO_NAME, win, win_smooth, N_EPISODES_EVAL, EPISODE_WINDOW, win_episodes
    ok, reward_now = computeMeanReward(LOG_DIR, 100)
    viz = Visdom(port=8097)
    win = timestepsPlot(viz, win, LOG_DIR, ENV_NAME, ALGO_NAME, bin_size=1, smooth=0, title=PLOT_TITLE, is_es=False)
    win_smooth = timestepsPlot(viz, win_smooth, LOG_DIR, ENV_NAME, ALGO_NAME, title=PLOT_TITLE + " smoothed",
                               is_es=False)
    win_episodes = episodePlot(viz, win_episodes, LOG_DIR, ENV_NAME,  ALGO_NAME, window=EPISODE_WINDOW,
                               title="PPO" + " [Episodes]", is_es=False)

    if reward_now > best_reward:
        print("New Record! Congrads!")
        print("Last Reward: {}, Best reward now: {}".format(best_reward, reward_now))
        best_reward = reward_now
    print("Last Reward: {}, Best reward now: {}".format(best_reward, reward_now))

def train(env_id, num_timesteps, seed, policy,
          n_envs=8, nminibatches=4, n_steps=128):
    """
    Train PPO2 model for atari environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    :param policy: (Object) The policy model to use (MLP, CNN, LSTM, ...)
    :param n_envs: (int) Number of parallel environments
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    """

    env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
    model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches,
                 lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
                 learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    model.learn(total_timesteps=num_timesteps, callback=callback)

    env.close()
    # Free memory
    del model

def folder_name(args):
    """
    return a folder name to log
    :param logdir:
    :return:
    """
    log_path = os.path.join(args.logdir, args.env)
    log_path = os.path.join(log_path,  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    return log_path

def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--logdir', default='logs/')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    global LOG_DIR, ENV_NAME
    ENV_NAME = args.env
    LOG_DIR = folder_name(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger.configure(folder=LOG_DIR)

    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy)


if __name__ == '__main__':
    main()
