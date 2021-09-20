import time
from collections import deque
import multiprocessing
import sys

import numpy as np
import tensorflow as tf

from copy import deepcopy
from rl_baselines.base_classes import StableBaselinesRLObject

from stable_baselines import PPO2
from stable_baselines.ppo2.ppo2 import get_schedule_fn, Runner,safe_mean
from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from srl_zoo.utils import   printGreen


class PPO2EWC(PPO2):
    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False):

        super(PPO2, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.ewc_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None

        self.Fisher_accum = None      # list of array
        self.pretrained_weight = None # list of array

        if _init_setup_model:
            self.setup_model()


    def load_weight(self):
        """
        new function that copy the value and the structure from self.params
        :return:
        """
        #Creation of a new variable to the class PPO2

        pretrained_weight =    [self.sess.run(var) for var in self.params]
        printGreen("Pretrained weight loaded")
        return pretrained_weight
    def compute_fisher_total(self, num_timesteps, runner ):
        """
               To get the diagonal of accumulated fisher information matrix
               :param num_timesteps: time steps for the sampling
               :param runner:
               :return:
               """

        num_samples = num_timesteps // self.n_batch

        # Creation of a new variable to the class PPO2
        self.Fisher_accum = [np.zeros_like(var) for var in self.pretrained_weight]

        F_total_accum = []
        for var in self.pretrained_weight:
            F_total_accum.append(np.zeros(shape=(np.prod(var.shape),np.prod(var.shape))))

        F_prev = deepcopy(self.Fisher_accum)
        mean_diffs = np.zeros(0)
        for iter in range(1, num_samples + 1):
            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
            # randomly sample  from the action, value and q-value
            step_ind = np.random.randint(self.n_steps)
            action_ind = tf.to_int32(tf.random.categorical(tf.log(self.train_model.policy_proba), 1))[:, 0]
            n_action = self.train_model.policy_proba.shape[1]
            action_mask = tf.one_hot(action_ind, depth=n_action, dtype=tf.bool, on_value=True, off_value=False)
            action_prob = tf.boolean_mask(self.train_model.policy_proba, action_mask)
            q_value = tf.boolean_mask(self.train_model.q_value, action_mask)
            # compute the fisher accumualated information
            for v in range(len(self.params)):
                # the first order derivative of the action proba by parameters (weight matrix)
                obs_sample = obs[step_ind:step_ind + 1]
                grad_action, grad_value, grad_q = self.sess.run([
                    tf.gradients(action_prob, self.params[v], unconnected_gradients='zero')[0],
                    tf.gradients(self.train_model._value, self.params[v], unconnected_gradients='zero')[0],
                    tf.gradients(q_value, self.params[v], unconnected_gradients='zero')[0]],
                    feed_dict={self.train_model.obs_ph: obs_sample,
                               self.params[v]: self.pretrained_weight[v]})


                """
                Add penalization only on the action space, or do the regularization on all outputs
                """
                if (len(np.unique(grad_action)) >1):
                    self.Fisher_accum[v] += np.square((grad_action + grad_value + grad_q))
                    # hessian_action = self.sess.run(
                    #     tf.hessians(action_prob, self.params[v])[0],
                    #     feed_dict={self.train_model.obs_ph: obs_sample,
                    #                self.params[v]: self.pretrained_weight[v]}
                    # )
                    #
                    # shape = np.prod(self.params[v].shape)
                    # hessian_action = np.reshape(hessian_action, [shape, shape])
                    grad_action_flat = grad_action.reshape(1,-1)
                    F_total_accum[v] += np.matmul(grad_action_flat.T , grad_action_flat)

            # Codes to show the convergence
            if (iter % (num_samples // 5) == 0):
                F_diff = 0
                Fisher_total = 0
                for v in range(len(self.Fisher_accum)):
                    F_diff += np.sum(np.absolute(self.Fisher_accum[v] / (iter + 1) - F_prev[v]))
                    Fisher_total += np.sum(np.absolute(self.Fisher_accum[v] / (iter + 1)))
                mean_diff = np.mean(F_diff)
                mean_diffs = np.append(mean_diffs, mean_diff)
                for v in range(len(self.Fisher_accum)):
                    F_prev[v] = self.Fisher_accum[v] / (iter + 1)
                printGreen(
                    "At iteration: {}, the new added information difference {}, total Fisher value {}".format(iter,
                                                                                                              F_diff,
                                                                                                              Fisher_total))

        printGreen("Fisher information computation complete")
        for v in range(len(self.Fisher_accum)):
            self.Fisher_accum[v] /= (num_samples)
            F_total_accum[v] /=num_samples
        #np.save('task_name.npy', self.Fisher_accum)
        np.save('CC_mt_total.npy',F_total_accum)




    def compute_fisher(self,num_timesteps,runner):
        """
        To get the diagonal of accumulated fisher information matrix
        :param num_timesteps: timesteps for the sampling
        :param runner:
        :return:
        """

        num_samples =num_timesteps // self.n_batch

        # Creation of a new variable to the class PPO2
        self.Fisher_accum = [np.zeros_like(var) for var in self.pretrained_weight]

        F_prev = deepcopy(self.Fisher_accum)
        mean_diffs = np.zeros(0)
        for iter in range(1, num_samples + 1):
            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
            #randomly sample  from the action, value and q-value
            step_ind = np.random.randint(self.n_steps)
            action_ind  = tf.to_int32(tf.random.categorical(tf.log(self.train_model.policy_proba), 1))[:, 0]
            n_action = self.train_model.policy_proba.shape[1]
            action_mask = tf.one_hot(action_ind ,depth =n_action ,dtype =tf.bool, on_value =True, off_value=False)
            action_prob = tf.boolean_mask(self.train_model.policy_proba, action_mask)
            q_value  = tf.boolean_mask(self.train_model.q_value ,action_mask)
            #compute the fisher accumualated information
            for v  in range(len(self.params)):
                #the first order derivative of the action proba by parameters (weight matrix)
                obs_sample  = obs[step_ind:step_ind+1]
                grad_action, grad_value, grad_q = self.sess.run([
                    tf.gradients(action_prob, self.params[v], unconnected_gradients='zero')[0],
                    tf.gradients(self.train_model._value, self.params[v], unconnected_gradients='zero')[0],
                    tf.gradients(q_value, self.params[v], unconnected_gradients='zero')[0]],
                    feed_dict={self.train_model.obs_ph: obs_sample,
                               self.params[v]: self.pretrained_weight[v]})
                """
                Add penalization only on the action space, or do the regularization on all outputs
                """
                #if (len(np.unique(grad_action)) >1):
                self.Fisher_accum[v] += np.square((grad_action + grad_value + grad_q))
            # Codes to show the convergence
            if(iter % (num_samples//10)==0):
                F_diff = 0
                Fisher_total = 0
                for v in range(len(self.Fisher_accum)):
                    F_diff += np.sum(np.absolute(self.Fisher_accum[v] / (iter + 1) - F_prev[v]))
                    Fisher_total += np.sum(np.absolute(self.Fisher_accum[v]/ (iter + 1)))
                mean_diff = np.mean(F_diff)
                mean_diffs = np.append(mean_diffs, mean_diff)
                for v in range(len(self.Fisher_accum)):
                    F_prev[v] = self.Fisher_accum[v] / (iter + 1)
                printGreen("At iteration: {}, the new added information difference {}, total Fisher value {}".format(iter,F_diff, Fisher_total))

        printGreen("Fisher information computation complete")
        for v in range(len(self.Fisher_accum)):
            self.Fisher_accum[v] /= (num_samples)


    def set_ewc_model(self, runner , num_timesteps=50000, ewc_weight = 400. ):
        """
        set up the model for ewc
        :param runner:
        :param num_timesteps: (int) the time steps for sampling the fisher information
        :param ewc_weight: (float) weight for ewc loss
        :return:
        """
        with self.graph.as_default():

            self.compute_fisher_total(num_timesteps, runner)
            with tf.variable_scope("ewc_loss", reuse=False):
                loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                # printGreen(self.sess.run(loss , {self.train_model.obs_ph: obs,  self.train_model.rewards_ph: reward}))
                self.ewc_loss = 0

                for v in range(len(self.params)):
                    self.ewc_loss += (ewc_weight / 2) * tf.reduce_sum(
                        tf.multiply(self.Fisher_accum[v], tf.square(self.params[v] - self.pretrained_weight[v])))
                # This is some codes for a naive regularization
                # for v in range(len(self.params)):
                #     self.ewc_loss += (ewc_weight / 2) * tf.reduce_sum(
                #          tf.square(self.params[v] - self.pretrained_weight[v]))
                loss += self.ewc_loss

                tf.summary.scalar('elastic_weight_loss', self.ewc_loss)

                grads = tf.gradients(loss, self.params)
                if self.max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._ewc_train = trainer.apply_gradients(grads)


            tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101
            self.summary = tf.summary.merge_all()
        self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, ewc=False):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._ewc_train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._ewc_train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, ewc_loss , value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.ewc_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._ewc_train], td_map)
            print(ewc_loss)
        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac


    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True ):

        self.pretrained_weight = self.load_weight()
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()


            self.set_ewc_model(runner)

            restores = []
            for param, loaded_p in zip(self.params, self.pretrained_weight):
                restores.append(param.assign(loaded_p))
            self.sess.run(restores)

            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            nupdates = total_timesteps // self.n_batch
            flag_ewc = False
            for update in range(1, nupdates + 1):
                assert self.n_batch % self.nminibatches == 0

                if(update> 8.e5//self.n_batch):
                    flag_ewc =True

                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / nupdates
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)
                # true_reward is the reward without discount

                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, writer=writer,
                                                                 update=timestep,ewc=flag_ewc))

                    self.num_timesteps += (self.n_batch * self.noptepochs) // batch_size * update_fac
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprangenow, *slices, update=timestep,
                                                                 writer=writer, states=mb_states ,ewc=flag_ewc))
                    self.num_timesteps += (self.n_envs * self.noptepochs) // envs_per_batch * update_fac
                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))


                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                      true_reward.reshape((self.n_envs, self.n_steps)),
                                                                      masks.reshape((self.n_envs, self.n_steps)),
                                                                      writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("nupdates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

            return self

class PPO2EWCModel(StableBaselinesRLObject):
    """
     object containing the interface between baselines.ppo2 and this code base
     PPO2: Proximal Policy Optimization (GPU Implementation)
     """

    LOG_INTERVAL = 10  # log RL model performance every 10 steps
    SAVE_INTERVAL = 1  # Save RL model every 1 steps

    def __init__(self):
        super(PPO2EWCModel, self).__init__(name="ppo2", model_class=PPO2EWC)


    def customArguments(self, parser):
        super().customArguments(parser)
        parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)

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

        assert not (self.policy in ['lstm', 'lnlstm', 'cnnlstm', 'cnnlnlstm'] and args.num_cpu % 4 != 0), \
            "Error: Reccurent policies must have num cpu at a multiple of 4."

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
                "cliprange": 0.2,
                "ewc_weight": 0
            }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
