import pdb
from pdb import set_trace
import numpy as np
import tensorflow as tf
import sys
import multiprocessing


from itertools import zip_longest
import warnings


from stable_baselines.common import tf_util, SetVerbosity
from stable_baselines.common.policies import nature_cnn, ActorCriticPolicy, LstmPolicy
from stable_baselines.a2c.utils import ortho_init
from stable_baselines import PPO2

from rl_baselines.progressive_nn.prnn_policy import ProgressiveMlpPolicy
from srl_zoo.utils import printRed,printGreen,printYellow


class ProgPPO2(PPO2):
    def __init__(self, policy, env, prev_cols=(),gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False):
        """
        Parameters could be referred to those of PPO2 in stable_baselines
        :param prev_cols: The previous pre-trained model, the progressive model
        """
        super(ProgPPO2, self).__init__(
            policy=policy, env=env, verbose=verbose, _init_setup_model=False, policy_kwargs=policy_kwargs)
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
        self.prev_cols = prev_cols

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.dict_res_tensor_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
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

        if _init_setup_model:
            self.setup_progressive_model()

    def setup_progressive_model(self):

        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None

                # TODO: update the prnn for lstm policy
                if issubclass(self.policy, LstmPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                tensor_name, _ = self.getResidualTensorName()

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step,
                                        prev_policy=[model.policy for model in self.prev_cols],
                                        prev_sess=[model.sess for model in self.prev_cols],
                                        reuse=False, **self.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True, custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              prev_policy=[model.train_model.policy for model in self.prev_cols],
                                              prev_sess=[model.sess for model in self.prev_cols],
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    #action placeholder
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    #advantages placeholder
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model._value
                    vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                        train_model._value - self.old_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leiber', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))

                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state

                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()


    def getResidualTensorName(self):
        res_tensor_name = []
        res_tensor_value = {}
        if(len(self.prev_cols)==0):
            return res_tensor_name, res_tensor_value
        graph = self.prev_cols[0].sess.graph
        all_operations = graph.get_operations()
        for ops in all_operations:
            name = ops.name
            if (("loss" not in name) and ( ("train_model/model/" in name))
                    and ("add" in name) and "res" not in name
                    ):
                res_tensor_name.append(name)
                res_tensor_value[name] = []
        return res_tensor_name, res_tensor_value

    """
    This part is very slow, should get quiker.
    """
    def residualBlock(self, obs):
        res_tensor_name, res_tensor_value = self.getResidualTensorName()

        for model in self.prev_cols:
            sess = model.sess
            prev_graph = sess.graph
            feed_dict = {prev_graph.get_operation_by_name('train_model/input/Ob').values(): obs}
            for name in res_tensor_name:
                tensor = prev_graph.get_operation_by_name(name).values()
                res_tensor_value[name].append((sess.run(tf.squeeze(tensor), feed_dict)))

        if(len(self.prev_cols)>0):
            for name in res_tensor_name:
                if(len(res_tensor_value[name][0].shape)==1):
                    batch_size = res_tensor_value[name][0].shape[0]
                    for i,arr in enumerate(res_tensor_value[name]):
                        res_tensor_value[name][i] = arr.reshape([batch_size,1])
                res_tensor_value[name] = np.concatenate(res_tensor_value[name],axis=1)
        return res_tensor_value, res_tensor_name

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None):
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
        res_tensor_value, res_tensor_name = self.residualBlock(obs)



        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}

        train_model_res_map = {}

        if(len(res_tensor_name)>0 ):
            for key in set(res_tensor_value.keys()).intersection(set(self.train_model.dict_res_tensor_ph.keys())):
                train_model_res_map[self.train_model.dict_res_tensor_ph[key]] = res_tensor_value[key]

            # train_model_res_map = {self.train_model.dict_res_tensor_ph[key]: res_tensor_value[key]
            #           for key in res_tensor_value.keys()}
        # act_model_res_map = {self.act_model.dict_res_tensor_ph[key]: res_tensor_value[key]
        #                      for key in self.act_model.dict_res_tensor_ph.keys()}

            td_map = {**td_map, **train_model_res_map}
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
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac