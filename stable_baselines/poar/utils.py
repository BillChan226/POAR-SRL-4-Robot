import tensorflow as tf

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from stable_baselines.a2c.utils import ortho_init,  conv_to_fc
import numpy as np
from ipdb import set_trace as tt
from srl_zoo.utils import printYellow, printGreen, printRed
from functools import partial

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def conv(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)


def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
    """
    Creates a fully connected layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden], initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(input_tensor, weight) + bias


def conv_t(input_tensor, scope, *, n_filters, filter_size, stride, output_shape=None,
           pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_filters, n_input]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d_transpose(input_tensor, weight, strides=strides, output_shape=output_shape,
                                             padding=pad, data_format=data_format)


def batch_norm(inputs, mode='NHWC'):
    """
    return a tensor that is batch-nomed
    :param inputs:
    :param mode: by default, channel last
    :return:
    """
    if mode == 'NHWC':
        norm_func = tf.layers.BatchNormalization(axis=-1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                 center=True, scale=True, fused=True)
    elif mode == 'NCHW':
        norm_func = tf.layers.BatchNormalization(axis=1,momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                 center=True, scale=True, fused=True)
    else:
        raise NotImplementedError
    return norm_func(inputs)


def batch_norm_relu_deconv(input_tensor, scope, *, n_filters, filter_size, stride, output_shape=None,
                           pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    a deconvolution with batch normalization and relu activation
    :param input_tensor:
    :param scope:
    :param n_filters:
    :param filter_size:
    :param stride:
    :param output_shape:
    :param pad:
    :param init_scale:
    :param data_format:
    :param one_dim_bias:
    :return:
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    channel_ax = 3
    strides = [1, stride, stride, 1]
    bshape = [1, 1, 1, n_filters]

    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_filters, n_input]

    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        d_conv_layer = bias + tf.nn.conv2d_transpose(input_tensor, weight, strides=strides, output_shape=output_shape,
                                             padding=pad, data_format=data_format)

        return tf.nn.relu(tf.compat.v1.layers.batch_normalization(d_conv_layer, axis=-1, momentum=_BATCH_NORM_DECAY,
                                                                  epsilon=_BATCH_NORM_EPSILON,
                                                                  center=True, scale=True, fused=True))

def bn_autoencoder(obs, state_dim):
    activation = tf.nn.relu
    e1 = activation(batch_norm(conv(scope='e1', input_tensor=obs, n_filters=64, filter_size=8, stride=4, pad='SAME')))
    e2 = activation(batch_norm(conv(scope='e2', input_tensor=e1, n_filters=64, filter_size=4, stride=2, pad='SAME')))
    e3 = activation(batch_norm(conv(scope='e3', input_tensor=e2, n_filters=64, filter_size=4, stride=2, pad='SAME')))

#    m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    m3_flat = conv_to_fc(e3)
    latent = linear(scope='latent', input_tensor=m3_flat, n_hidden=state_dim)

    d0 = tf.reshape(linear(scope='deocder_fc', input_tensor=latent, n_hidden=m3_flat.get_shape()[-1]), shape=tf.shape(e3))
    d1 = activation(batch_norm(tf.layers.conv2d_transpose(d0, filters=64, kernel_size=4, strides=2, padding='same')))
    d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=4, strides=2, padding='same')))
    d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=8, strides=4, padding='same')))
    output = tf.nn.sigmoid(conv(scope='reconstruction', input_tensor=d3, n_filters=3, filter_size=3, stride=1, pad='SAME'))
    return output, latent

def autoencoderMP(obs, state_dim):
    activation = tf.nn.relu
    e1 = activation(batch_norm(conv(scope='e1', input_tensor=obs, n_filters=64, filter_size=3, stride=1, pad='SAME')))
    m1 = tf.layers.max_pooling2d(e1, pool_size=2, strides=2)
    e2 = activation(batch_norm(conv(scope='e2', input_tensor=m1, n_filters=64, filter_size=3, stride=1, pad='SAME')))
    m2 = tf.layers.max_pooling2d(e2, pool_size=2, strides=2)
    e3 = activation(batch_norm(conv(scope='e3', input_tensor=m2, n_filters=64, filter_size=3, stride=1, pad='SAME')))
    m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    e4 = activation(batch_norm(conv(scope='e4', input_tensor=m3, n_filters=64, filter_size=3, stride=1, pad='SAME')))
    m4 = tf.layers.max_pooling2d(e4, pool_size=2, strides=2)
    m3_flat = conv_to_fc(m4)
    latent = m3_flat
    #latent = linear(scope='latent', input_tensor=m3_flat, n_hidden=state_dim)

    #d0 =tf.reshape(linear(scope='deocder_fc', input_tensor=latent, n_hidden=m3_flat.get_shape()[-1]), shape=tf.shape(m3))
    d1 = activation(batch_norm(tf.layers.conv2d_transpose(m4, filters=128, kernel_size=3, strides=2, padding='same')))
    d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=128, kernel_size=3, strides=2, padding='same')))
    d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=256, kernel_size=3, strides=2, padding='same')))
    d4 = activation(batch_norm(tf.layers.conv2d_transpose(d3, filters=256, kernel_size=3, strides=2, padding='same')))
    output = tf.nn.sigmoid(conv(scope='reconstruction', input_tensor=d4, n_filters=3, filter_size=3, stride=1, pad='SAME'))
    return output, latent

def nature_autoencoder(obs, state_dim, reuse=tf.AUTO_REUSE):
    """
    The autoencoder structure that has encoder similar to the original structure in PPO2
    :param obs:
    :param state_dim:
    :param reuse:
    :return:
    """
    activation = tf.nn.relu
    with tf.variable_scope('encoder1', reuse=reuse):
        e1 = activation(conv(scope='conv2d', input_tensor=obs, n_filters=64, filter_size=8, stride=4, pad='SAME'))
    with tf.variable_scope('encoder2', reuse=reuse):
        e2 = activation(conv(scope='conv2d', input_tensor=e1, n_filters=64, filter_size=4, stride=2, pad='SAME'))
    with tf.variable_scope('encoder3', reuse=reuse):
        e3 = activation(conv(scope='conv2d', input_tensor=e2, n_filters=64, filter_size=3, stride=1, pad='SAME'))

    #m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    with tf.variable_scope('latent_observation', reuse=tf.AUTO_REUSE):
        m3_flat = conv_to_fc(e3)
        latent = linear(scope='latent', input_tensor=m3_flat, n_hidden=state_dim)

    with tf.variable_scope('decoder_fc', reuse=reuse):
        d0 =tf.reshape(linear(scope='mlp', input_tensor=latent, n_hidden=m3_flat.get_shape()[-1]), shape=tf.shape(e3))
    with tf.variable_scope('decoder1', reuse=reuse):
        d1 = activation(batch_norm(tf.layers.conv2d_transpose(d0, filters=64, kernel_size=3, strides=1, padding='same')))
    with tf.variable_scope('decoder2', reuse=reuse):
        d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=4, strides=2, padding='same')))
    with tf.variable_scope('decoder3', reuse=reuse):
        d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=8, strides=4, padding='same')))
    with tf.variable_scope('reconstruction', reuse=reuse):
        output = tf.nn.tanh(conv(scope='conv2d', input_tensor=d3, n_filters=3, filter_size=3, stride=1, pad='SAME'))
        return output, latent


def natural_autoencoder(obs, state_dim, reuse=tf.AUTO_REUSE):
    """
    The autoencoder structure that has encoder similar to the original structure in PPO2
    :param obs:
    :param state_dim:
    :param reuse:
    :return:
    """
    activation = tf.nn.relu
    with tf.variable_scope('encoder1', reuse=reuse):
        e1 = activation(conv(scope='conv2d', input_tensor=obs, n_filters=64, filter_size=8, stride=4, pad='SAME'))
    with tf.variable_scope('encoder2', reuse=reuse):
        e2 = activation(conv(scope='conv2d', input_tensor=e1, n_filters=64, filter_size=4, stride=2, pad='SAME'))
    with tf.variable_scope('encoder3', reuse=reuse):
        e3 = activation(conv(scope='conv2d', input_tensor=e2, n_filters=64, filter_size=3, stride=1, pad='SAME'))

    #m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    with tf.variable_scope('latent_observation', reuse=tf.AUTO_REUSE):
        m3_flat = conv_to_fc(e3)
        latent = linear(scope='latent', input_tensor=m3_flat, n_hidden=state_dim)

    with tf.variable_scope('decoder_fc', reuse=reuse):
        d0 =tf.reshape(linear(scope='mlp', input_tensor=latent, n_hidden=m3_flat.get_shape()[-1]), shape=tf.shape(e3))
    with tf.variable_scope('decoder1', reuse=reuse):
        d1 = activation(batch_norm(tf.layers.conv2d_transpose(d0, filters=64, kernel_size=3, strides=1, padding='same')))
    with tf.variable_scope('decoder2', reuse=reuse):
        d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=4, strides=2, padding='same')))
    with tf.variable_scope('decoder3', reuse=reuse):
        d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=8, strides=4, padding='same')))
    with tf.variable_scope('reconstruction', reuse=reuse):
        output = tf.nn.tanh(conv(scope='conv2d', input_tensor=d3, n_filters=4, filter_size=5, stride=1, pad='VALID'))
        return output, latent

def encoder(obs, state_dim, reuse=tf.AUTO_REUSE):
    activation = tf.nn.relu
    with tf.variable_scope('encoder1', reuse=reuse):
        e1 = activation(conv(scope='conv2d', input_tensor=obs, n_filters=64, filter_size=8, stride=4, pad='SAME'))
    with tf.variable_scope('encoder2', reuse=reuse):
        e2 = activation(conv(scope='conv2d', input_tensor=e1, n_filters=64, filter_size=4, stride=2, pad='SAME'))
    with tf.variable_scope('encoder3', reuse=reuse):
        e3 = activation(conv(scope='conv2d', input_tensor=e2, n_filters=64, filter_size=3, stride=1, pad='SAME'))

    #m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    with tf.variable_scope('latent_observation', reuse=tf.AUTO_REUSE):
        m3_flat = conv_to_fc(e3)
        fc1 = linear(scope='fc1', input_tensor=m3_flat, n_hidden=256)
        fc2 = linear(scope='fc2', input_tensor=fc1, n_hidden=128)
        fc3 = linear(scope='fc3', input_tensor=fc2, n_hidden=64)
        latent = linear(scope='latent', input_tensor=fc3, n_hidden=state_dim)
    return obs, latent

def naive_autoencoder(obs, state_dim, reuse=tf.AUTO_REUSE):
    activation = tf.nn.relu
    with tf.variable_scope('encoder1', reuse=reuse):
        e1 = activation(conv(scope='conv2d', input_tensor=obs, n_filters=64, filter_size=8, stride=4, pad='SAME'))
    with tf.variable_scope('encoder2', reuse=reuse):
        e2 = activation(conv(scope='conv2d', input_tensor=e1, n_filters=64, filter_size=4, stride=2, pad='SAME'))
    with tf.variable_scope('encoder3', reuse=reuse):
        e3 = activation(conv(scope='conv2d', input_tensor=e2, n_filters=64, filter_size=3, stride=1, pad='SAME'))

    #m3 = tf.layers.max_pooling2d(e3, pool_size=2, strides=2)
    with tf.variable_scope('latent_observation', reuse=tf.AUTO_REUSE):
        m3_flat = conv_to_fc(e3)
        latent = m3_flat

    with tf.variable_scope('decoder1', reuse=reuse):
        d1 = activation(batch_norm(tf.layers.conv2d_transpose(e3, filters=64, kernel_size=3, strides=2, padding='same')))
    with tf.variable_scope('decoder2', reuse=reuse):
        d2 = activation(batch_norm(tf.layers.conv2d_transpose(d1, filters=64, kernel_size=3, strides=2, padding='same')))
    with tf.variable_scope('decoder3', reuse=reuse):
        d3 = activation(batch_norm(tf.layers.conv2d_transpose(d2, filters=64, kernel_size=3, strides=2, padding='same')))
    with tf.variable_scope('reconstruction', reuse=reuse):
        output = tf.nn.tanh(conv(scope='conv2d', input_tensor=d3, n_filters=3, filter_size=3, stride=1, pad='SAME'))
        return output, latent

def inverse_net(state, next_state, ac_space):
    """
    return the prediction of the action
    :param state:
    :param next_state:
    :param ac_space:
    :return:
    """
    activation = tf.nn.relu
    with tf.variable_scope("inverse"):
        concat_state = tf.concat([state, next_state], axis=1, name='concat_state')
        layer1 = activation(linear(input_tensor=concat_state, scope='fc1', n_hidden=64, init_scale=np.sqrt(2)))
        layer2 = activation(linear(input_tensor=layer1, scope='fc2', n_hidden=64, init_scale=np.sqrt(2)))
        if isinstance(ac_space, Box):  # TODO: for the continuous action
            return linear(input_tensor=layer2, scope='srl_action', n_hidden=ac_space.shape)
        else:  # discrete action
            return tf.nn.softmax(linear(input_tensor=layer2, scope='srl_action', n_hidden=ac_space.n))

def forward_net(state, action, ac_space, state_dim=512):
    """
    predict next state with the current state and the action
    :param state:
    :param action:
    :param ac_space:
    :param state_dim:
    :return:
    """
    activation = tf.nn.relu
    with tf.variable_scope("forward"):
        if isinstance(ac_space, Box):
            concat_state_action = tf.concat([state, action], axis=1, name='state_action')
        else:
            concat_state_action = tf.concat([state, tf.one_hot(action, ac_space.n)], axis=1, name='state_action')
        layer1 = activation(linear(input_tensor=concat_state_action, scope='fc1', n_hidden=64, init_scale=np.sqrt(2)))
        layer2 = activation(linear(input_tensor=layer1, scope='fc2', n_hidden=64, init_scale=np.sqrt(2)))
        return linear(input_tensor=layer2, scope='srl_state', n_hidden=state_dim)


def transition_net(state, action, ac_space, state_dim=512):
    """
    predict the change of the state, s_{t+1} = s_t + transition
    :param state:
    :param action:
    :param ac_space:
    :param state_dim:
    :return:
    """
    activation = tf.nn.relu

    with tf.variable_scope("transition"):
        if isinstance(ac_space, Box):
            concat_state_action = tf.concat([state, action], axis=1, name='state_action')
        else:
            concat_state_action = tf.concat([state, tf.one_hot(action, ac_space.n)], axis=1, name='state_action')
        layer1 = activation(linear(input_tensor=concat_state_action, scope='fc1', n_hidden=64, init_scale=np.sqrt(2)))
        layer2 = activation(linear(input_tensor=layer1, scope='fc2', n_hidden=64, init_scale=np.sqrt(2)))
        return linear(input_tensor=layer2, scope='srl_state', n_hidden=state_dim)


def reward_net(state, next_state, reward_dim=1):
    """
    To predict the reward, this network is used for a classification problem, a dense reward should be implented further
    :param state:
    :param next_state:
    :param reward_dim:
    :return:
    """
    activation = tf.nn.relu

    with tf.variable_scope("reward"):
        concat_state = tf.concat([state, next_state], axis=1, name='concat_state')
        layer1 = activation(linear(input_tensor=concat_state, scope='fc1', n_hidden=64, init_scale=np.sqrt(2)))
        layer2 = activation(linear(input_tensor=layer1, scope='fc2', n_hidden=64, init_scale=np.sqrt(2)))
        return linear(input_tensor=layer2, scope='srl_state', n_hidden=reward_dim)

def pca(data, dim=2):
    # preprocess the data
    X = data
    X_mean = np.mean(X, axis=0)
    X = X - X_mean
    # svd
    U, S, V = np.linalg.svd(X.T)
    C = np.matmul(X,U[:,:dim])
    return C

# ==============================================================================
"""Domain Adaptation Loss Functions.
The following domain adaptation loss functions are defined:
- Maximum Mean Discrepancy (MMD).
  Relevant paper:
    Gretton, Arthur, et al.,
    "A kernel two-sample test."
    The Journal of Machine Learning Research, 2012
- Correlation Loss on a batch.
"""

################################################################################
# SIMILARITY LOSS
################################################################################
def maximum_mean_discrepancy(x, y, kernel):
    """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

def mmd_loss(source_samples, target_samples, weight=1, scope=None):
    """Adds a similarity loss term, the MMD between two representations.
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    Args:
      source_samples: a tensor of shape [num_samples, num_features].
      target_samples: a tensor of shape [num_samples, num_features].
      weight: the weight of the MMD loss.
      scope: optional name scope for summary tags.
    Returns:
      a scalar tensor representing the MMD loss value.
    """
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight
    assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
    with tf.control_dependencies([assert_op]):
        tag = 'MMD Loss'
        if scope:
            tag = scope + tag
        tf.summary.scalar(tag, loss_value)
        #tf.losses.add_loss(loss_value)
    print('loss_value!!!!!!!!', loss_value)

    return loss_value

def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)


    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        print("source_shape", x.get_shape())
        print("target_shape", y.get_shape())
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    # x will be num_samples x num_features x 1,
    # and y will be 1 x num_features x num_samples (after broadcasting).
    # After the substraction we will get a
    # num_x_samples x num_features x num_y_samples matrix.
    # The resulting dist will be of shape num_y_samples x num_x_samples.
    # and thus we need to transpose it again.
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))