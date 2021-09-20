import warnings
from itertools import zip_longest


from abc import abstractmethod
import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.input import observation_input
from stable_baselines.poar.utils import nature_autoencoder, bn_autoencoder, inverse_net, \
    forward_net, autoencoderMP, naive_autoencoder, transition_net, encoder, reward_net, natural_autoencoder
from gym.spaces.discrete import Discrete

from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution

from ipdb import set_trace as tt

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value


class SRLActorCriticPolicy(BasePolicy):
    """
    Policy object that implements actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, split_dim=(), reuse=False, scale=False):
        super(SRLActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                   scale=scale)
        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None
        self.reconstruct_obs = None
        self.next_reconstruct_obs = None
        self.next_processed_obs = None
        self.srl_action = None
        self.srl_state = None
        self.next_latent_obs = None
        self.latent_obs = None
        self.srl_reward = None
        self._split_dim = split_dim

    def _setup_init(self):
        """
        sets up the distibutions, actions, and value
        """
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                      for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            self._value_flat = self.value_fn[:, 0]

    @property
    def split_dim(self):
        return self._split_dim

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic actions."""
        return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution. Depends on pdtype."""
        return self._policy_proba

    @abstractmethod
    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
        raise NotImplementedError


class FeedForwardPolicy(SRLActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, feature_extraction="cnn", structure=None, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)
        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]
        with tf.variable_scope("model", reuse=reuse):
            # By default, we consider the inputs are raw_pixels
            if structure == 'autoencoder':
                self.reconstruct_obs, latent_obs = nature_autoencoder(self.processed_obs, state_dim=512)
                pi_latent = vf_latent = latent_obs
            elif structure == 'autoencoder_bn':
                self.reconstruct_obs, latent_obs = bn_autoencoder(self.processed_obs, state_dim=512)
                pi_latent = vf_latent = latent_obs
            elif structure == 'autoencoder_mlp':
                self.reconstruct_obs, latent_obs = nature_autoencoder(self.processed_obs, state_dim=512)
                pi_latent, vf_latent = mlp_extractor(latent_obs, net_arch, act_fun)
            else:
                self.reconstruct_obs = self.processed_obs
                pi_latent = vf_latent = nature_cnn(self.processed_obs, **kwargs)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class NatureCnnPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(NatureCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                              feature_extraction="cnn", **_kwargs)


class AEPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(AEPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                       feature_extraction='cnn', structure='autoencoder', **_kwargs)


class AEBNPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(AEBNPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                         feature_extraction='cnn', structure='autoencoder_bn', **_kwargs)


class AEMlpPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(AEMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction='cnn', structure='autoencoder_mlp', **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class SRLPolicy(SRLActorCriticPolicy):
    """
    SRL policy that can combine or split dimension for SRL model. An implementation of SRL in tensorflow
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, split_dim=200, feature_extraction="cnn", structure='autoencoder', **kwargs):
        super(SRLPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                        scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)

        # placeholder for the next observation
        with tf.variable_scope('Next_observation'):
            self.next_obs_ph, self.next_processed_obs = observation_input(ob_space, n_batch, name='NextObs',
                                                                          scale=(feature_extraction == "cnn"))
            self._action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape, name="action_ph")

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)
        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            # preserved for further use, for example, pi_latent could be passed in to a MLP strutre
            net_arch = [dict(vf=layers, pi=layers)]
        with tf.variable_scope("model", reuse=reuse):
            # By default, we consider the inputs are raw_pixels
            pi_latent = vf_latent = self.srl_scope(split_dim, ac_space, structure)
            # latent_obs = self.srl_scope(split_dim, ac_space, structure)
            # pi_latent, vf_latent = mlp_extractor(latent_obs, net_arch, act_fun)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def srl_scope(self, split_dim, ac_space, structure='autoencoder'):
        """
        The manager method that include all SRL models
        :param split_dim: dict with indicated dimension for each model
        :param ac_space:
        :param structure: the autoencoder structure
        :return:
        """
        if structure == 'autoencoder':
            encoder_fn = nature_autoencoder
        elif structure == 'natural':
            encoder_fn = natural_autoencoder
        elif structure == 'naive_autoencoder':
            encoder_fn = naive_autoencoder
        else:
            encoder_fn = encoder
        state_dim = 0
        previous_dim = 0
        dim_attr_dict = {}
        for key in split_dim:
            if split_dim[key] < 0:
                assert state_dim != previous_dim, "Error: the first dimension should not be 0"
                dim_attr_dict[key] = (previous_dim, state_dim)
            else:
                previous_dim = state_dim
                state_dim += split_dim[key]
                dim_attr_dict[key] = (previous_dim, state_dim)
        if 'domain' in split_dim:
            self.domain_dim = dim_attr_dict['domain']
        self.next_reconstruct_obs, self.next_latent_obs = encoder_fn(self.next_processed_obs, state_dim=state_dim)
        self.reconstruct_obs, self.latent_obs = encoder_fn(self.processed_obs, state_dim=state_dim)
        # We make next_latent_obs to be observable from outside to compute the loss with srl_state

        with tf.variable_scope('SRL'):
            if "forward" in split_dim:
                self.srl_state = forward_net(
                    self.latent_obs[..., dim_attr_dict["forward"][0]:dim_attr_dict["forward"][1]],
                    self._action_ph, ac_space, state_dim=state_dim)
            if "inverse" in split_dim:
                self.srl_action = inverse_net(
                    self.latent_obs[..., dim_attr_dict["inverse"][0]:dim_attr_dict["inverse"][1]],
                    self.next_latent_obs[..., dim_attr_dict["inverse"][0]:dim_attr_dict["inverse"][1]], ac_space)
            if "reward" in split_dim:
                # if dim_attr_dict["reward"][1] - dim_attr_dict["reward"][0] == dim_attr_dict["inverse"][1]-dim_attr_dict["inverse"][0]:
                #     self.srl_reward = reward_net(
                #         self.latent_obs[..., dim_attr_dict["reward"][0]:dim_attr_dict["reward"][1]] -
                #         self.latent_obs[..., dim_attr_dict["inverse"][0]:dim_attr_dict["inverse"][1]],
                #         self.next_latent_obs[..., dim_attr_dict["reward"][0]:dim_attr_dict["reward"][1]] -
                #         self.next_latent_obs[..., dim_attr_dict["inverse"][0]:dim_attr_dict["inverse"][1]],
                #         reward_dim=2
                #     )
                # else:
                self.srl_reward = reward_net(
                    self.latent_obs[..., dim_attr_dict["reward"][0]:dim_attr_dict["reward"][1]],
                    self.next_latent_obs[..., dim_attr_dict["reward"][0]:dim_attr_dict["reward"][1]], reward_dim=10)
        return self.latent_obs

    def step(self, obs, next_obs=None, state=None, mask=None, deterministic=False):
        """
        for the model to take action, predict value function
        :param obs:
        :param next_obs:
        :param state:
        :param mask:
        :param deterministic:
        :return:
        """
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})

        p_obs, ae_obs, latent = self.sess.run([self.processed_obs, self.reconstruct_obs, self.latent_obs],
                                              {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp, ae_obs, latent

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class NAEPolicy(SRLPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(NAEPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction='cnn', structure='naive_autoencoder', **_kwargs)


class AESRLPolicy(SRLPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(AESRLPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction='cnn', structure='autoencoder', **_kwargs)


class NAESRLPolicy(SRLPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(NAESRLPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction='cnn', structure='natural', **_kwargs)

