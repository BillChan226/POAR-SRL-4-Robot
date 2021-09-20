from stable_baselines import POAR
from stable_baselines import POAR_MMD

from rl_baselines.base_classes import StableBaselinesRLObject
from collections import OrderedDict
from srl_zoo.utils import parseLossArguments


class POARModel(StableBaselinesRLObject):
    """
    object containing the interface between baselines.ppo2 and this code base
    PPO2: Proximal Policy Optimization (GPU Implementation)
    """

    LOG_INTERVAL = 10  # log RL model performance every 10 steps
    SAVE_INTERVAL = 1  # Save RL model every 1 steps

    def __init__(self):
        super(POARModel, self).__init__(name="poar", model_class=POAR_MMD)


    def customArguments(self, parser):
        super().customArguments(parser)
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
        parser.add_argument('--structure', type=str, default='srl', help='The structure for poar')
        parser.add_argument('--losses', nargs='+', default=["autoencoder"], **parseLossArguments(
            choices=["forward", "inverse", "reward", "entropy", "autoencoder", "domain"],
            help='The wanted losses. One may also want to specify a weight and dimension '
                 'that apply as follows: "<name>:<weight>:<dimension>".'))
        return parser

    @classmethod
    def getOptParam(cls):
        return {
            "lam": (float, (0, 1)),
            "gamma": (float, (0, 1)),
            "max_grad_norm": (float, (0, 1)),
            "vf_coef": (float, (0, 1)),
            "learning_rate": (float, (1e-2, 1e-5)),
            "ent_coef": (float, (0, 1)),
            "cliprange": (float, (0, 1)),
            "noptepochs": (int, (1, 10)),
            "n_steps": (int, (32, 2048))
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}
        losses_weights_dict = OrderedDict()
        split_dimensions = OrderedDict()
        for loss, weight, split_dim in args.losses:
            losses_weights_dict[loss] = weight
            split_dimensions[loss] = split_dim
        param_kwargs = {
            "verbose": 1,
            "n_steps": 128,
            "ent_coef": 0.01,
            "learning_rate": lambda f: f * 2.5e-4,
            "srl_lr": lambda f: f * 5e-4,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "gamma": 0.99,
            "lam": 0.95,
            "nminibatches": 4,
            "noptepochs": 4,
            "cliprange": 0.2,
            "split_dim": split_dimensions,
            "srl_weight": losses_weights_dict,
            "demo_path": args.demon_path,
            "state_graph": args.state_graph
        }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
