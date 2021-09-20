import pdb
import json
import yaml
from datetime import datetime
import pdb


from state_representation.registry import registered_srl
from state_representation import SRLType
from rl_baselines.base_classes import StableBaselinesRLObject
from rl_baselines.progressive_nn.utils import *
from rl_baselines import AlgoType, ActionType
from rl_baselines.progressive_nn.ppo2_model import ProgPPO2
from rl_baselines.rl_algorithm.ppo2 import PPO2Model

def loadConfigAndSetup(log_dir):
    """
    load training variable from a pre-trained model
    :param log_dir: the path where the model is located,
    example: logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-05-07_11h32_39
    :return: train_args, algo_name, algo_class(stable_baselines.PPO2), srl_model_path, env_kwargs
    """
    assert "ppo2" in log_dir or "prnn" in log_dir
    if ("ppo2" in log_dir):
        algo_name = "ppo2"
        algo_class = PPO2Model
    elif ("prnn" in log_dir):
        algo_name = "prnn"
        algo_class = ProgressiveNN
    if log_dir[-3:] != 'pkl':
        load_path = "{}/{}.pkl".format(log_dir, algo_name)
    else:
        load_path = log_dir
        log_dir = os.path.dirname(load_path)+'/'

    env_globals = json.load(open(log_dir + "env_globals.json", 'r'))
    train_args = json.load(open(log_dir + "args.json", 'r'))
    env_kwargs = {
        "renders":False,
        "shape_reward": train_args["shape_reward"],
        "action_joints": train_args["action_joints"],
        "is_discrete": not train_args["continuous_actions"],
        "random_target": train_args.get('random_target', False),
        "srl_model": train_args["srl_model"]
    }

    # load it, if it was defined
    if "action_repeat" in env_globals:
        env_kwargs["action_repeat"] = env_globals['action_repeat']

    # Remove up action
    if train_args["env"] == "Kuka2ButtonGymEnv-v0":
        env_kwargs["force_down"] = env_globals.get('force_down', True)
    else:
        env_kwargs["force_down"] = env_globals.get('force_down', False)

    if train_args["env"] == "OmnirobotEnv-v0":
        env_kwargs["simple_continual_target"] = env_globals.get("simple_continual_target", False)
        env_kwargs["circular_continual_move"] = env_globals.get("circular_continual_move", False)
        env_kwargs["square_continual_move"] = env_globals.get("square_continual_move", False)
        env_kwargs["eight_continual_move"] = env_globals.get("eight_continual_move", False)

    srl_model_path = None
    if train_args["srl_model"] != "raw_pixels":
        train_args["policy"] = "mlp"
        path = env_globals.get('srl_model_path')

        if path is not None:
            env_kwargs["use_srl"] = True
            # Check that the srl saved model exists on the disk
            assert os.path.isfile(env_globals['srl_model_path']), "{} does not exist".format(env_globals['srl_model_path'])
            srl_model_path = env_globals['srl_model_path']
            env_kwargs["srl_model_path"] = srl_model_path
    return algo_class, train_args, load_path, srl_model_path, env_kwargs, algo_name








class ProgressiveNN(StableBaselinesRLObject):
    """
     object containing the interface between baselines.ppo2 and this code base
     PPO2: Proximal Policy Optimization (GPU Implementation)
     """

    LOG_INTERVAL = 10  # log RL model performance every 10 steps
    SAVE_INTERVAL = 1  # Save RL model every 1 steps

    def __init__(self):
        super(ProgressiveNN, self).__init__(name="prnn", model_class=ProgPPO2)


    def customArguments(self, parser):
        super().customArguments(parser)
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)

        return parser


    def modelsLoader(self, previous_model_path,args):
        """
        To load previous pre-trained models
        :param previous_model_path: (list) list of paths to the model
        :param args: (namespace) usage for the num-cpu
        :return:
        """
        previous_models = []
        for path in previous_model_path:
            algo_class, train_args, load_path, srl_model_path, env_kwargs, algo_name = loadConfigAndSetup(path)
            log_dir ="/tmp/gym/test/"+"{}/{}/".format(algo_name, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
            os.makedirs(log_dir, exist_ok=True)

            train_args["num_cpu"]= args.num_cpu
            algo_args = type('attrib_dict', (), train_args)()
            #load the environment
            # Question: if we load the model, we should load which environment, the current one or the previous one???
            # If current one, may the model forget the previous task?
            envs = self.makeEnv(algo_args, env_kwargs, load_path_normalise=path)
            model = self.model_class.load(load_path,envs)
            previous_models.append(model)
        return tuple(previous_models)


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

        assert not (self.policy in ['lstm', 'lnlstm', 'cnnlstm', 'cnnlnlstm'] and args.num_cpu % 4 != 0), \
            "Error: Reccurent policies must have num cpu at a multiple of 4."



        #TODO the load path 就是load rl model path
        #Pass the pretrained model weight as some parameters
        previous_model_path = args.previous_models
        previous_models = self.modelsLoader(previous_model_path,args)


        if "lstm" in args.policy:
            param_kwargs = {
                "verbose": 1,
                "n_steps": 609,
                "ent_coef": 0.06415865069774951,
                "learning_rate": 0.004923676735761618,
                "vf_coef": 0.056219345567007695,
                "max_grad_norm": 0.19232704980689763,
                "gamma": 0.9752388470759489,
                "lam": 0.3987544314875193,
                "nminibatches": 4,
                "noptepochs": 8
            }
        else:
            param_kwargs = {
                "verbose": 1,
                "n_steps": 128,
                "ent_coef": 0.01,
                "learning_rate": lambda f: f * 2.5e-4,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gamma": 0.99,
                "lam": 0.95,
                "nminibatches": 4,
                "noptepochs": 4,
                "cliprange": 0.2
            }
        previous_cols = {"prev_cols": previous_models}
        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs, **previous_cols})