#!/usr/bin/python3

# Copyright 2017 Andres Milioto. All Rights Reserved.
#
#  This file is part of Bonnet.
#
#  Bonnet is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Bonnet is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Bonnet. If not, see <http://www.gnu.org/licenses/>.

'''
  Layer class, containing:
    - Definition of important layers
'''
import tensorflow as tf


def weight_variable(shape, train):
  print("W: ", shape, "Train:", train)
  return tf.get_variable("w", shape=shape,
                         initializer=tf.variance_scaling_initializer,
                         trainable=train)


def bias_variable(shape, train):
  print("b: ", shape, "Train:", train)
  return tf.get_variable("b", shape=shape,
                         initializer=tf.random_normal_initializer,
                         trainable=train)


def conv2d(x, W, stride=1, data_format="NCHW"):
  # default to a stride of 1 because it is the one we use the most
  if data_format == "NCHW":
    output = tf.nn.conv2d(x, W,
                          strides=[1, 1, stride, stride],
                          padding='SAME', data_format="NCHW")
  else:
    output = tf.nn.conv2d(x, W,
                          strides=[1, stride, stride, 1],
                          padding='SAME', data_format="NHWC")
  return output


def max_pool(x, stride=2, data_format="NCHW"):
  # default to a stride of 2 because it is the one we use the most
  if data_format == "NCHW":
    output = tf.nn.max_pool(x,
                            ksize=[1, 1, stride, stride],
                            strides=[1, 1, stride, stride],
                            padding='SAME', data_format="NCHW")
  else:
    output = tf.nn.max_pool(x,
                            ksize=[1, stride, stride, 1],
                            strides=[1, stride, stride, 1],
                            padding='SAME', data_format="NHWC")
  return output


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.variable_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.variable_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# definition for a full conv layer (variables+conv+relu+pool)


def conv_layer(input_tensor, kernel_nr, kernel_size, stride,
               train, summary=False, bnorm=True, relu=True,
               data_format="NCHW", bn_decay=0.95):
  """Builds a full conv layer, with variables and relu
  Args:
    input_tensor: input tensor
    kernel_nr: This layer's number of filters
    kernel_size: Size of the kernel [h, w]
    train: If we want to train this layer or not
    bnorm: Use batchnorm?
    relu: Use relu?
    data_format: Self explanatory
  Returns:
    output: Output tensor from the convolution
  """
  # get previous depth from input tensor
  if data_format == "NCHW":
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[1]
  else:
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[3]

  with tf.variable_scope('weights'):
    W = weight_variable([kernel_size[0], kernel_size[1],
                         prev_depth, kernel_nr], train)
    if summary:
      variable_summaries(W)

  with tf.variable_scope('convolution'):
    preactivations = conv2d(input_tensor, W, stride, data_format=data_format)
    if summary:
      variable_summaries(preactivations)

  if bnorm:
    with tf.variable_scope('batchnorm'):
      normalized = tf.contrib.layers.batch_norm(preactivations,
                                                center=True, scale=True,
                                                is_training=train,
                                                data_format=data_format,
                                                fused=True,
                                                decay=bn_decay,
                                                updates_collections=None)
      if summary:
        variable_summaries(normalized)
  else:
    normalized = preactivations

  if relu:
    with tf.variable_scope('relu'):
      output = relu = tf.nn.relu(normalized)
      if summary:
        variable_summaries(relu)
  else:
    output = normalized

  return output


def upsample_layer(input_tensor, train, upsample_factor=2, kernels=-1, data_format="NCHW"):
  """Builds a full conv layer, with variables and relu
  Args:
    input_tensor: input tensor
    upsample_factor: how much to upsample
    kernels: -1 = same as input, otherwise number of kernels to upsample
    data_format: Self explanatory
  Returns:
    output: Output tensor from the upsampling
  """
  if data_format == "NCHW":
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[1]
  else:
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[3]

  if kernels < 0:
    kernel_nr = prev_depth
  else:
    kernel_nr = kernels

  with tf.variable_scope('upconv'):
    output = tf.contrib.layers.conv2d_transpose(input_tensor,
                                                kernel_nr,
                                                (2, 2),
                                                stride=2,
                                                padding='VALID',
                                                data_format=data_format,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.variance_scaling_initializer,
                                                weights_regularizer=None,
                                                trainable=train)
  return output


def asym_conv_layer(input_tensor, kernel_nr, kernel_size, train, summary=False, bnorm=True, relu=True, data_format="NCHW"):
  """Builds a full asymetric conv layer, with variables and relu.
  Args:
    input_tensor: input tensor
    kernel_nr: This layer's number of filters
    kernel_size: Size of the kernel [symetric]
    train: ;If we want to train this layer or not
    bnorm: Use batchnorm?
    relu: Use relu?
    data_format: Self explanatory
  Returns:
    output: Output tensor from the convolution
  """
  with tf.variable_scope('horiz'):
    conv = conv_layer(input_tensor, kernel_nr, [kernel_size, 1], 1, train,
                      summary=summary, bnorm=bnorm, relu=relu, data_format=data_format)
  with tf.variable_scope('vert'):
    output = conv_layer(conv, kernel_nr, [1, kernel_size], 1, train,
                        summary=summary, bnorm=bnorm, relu=relu, data_format=data_format)

  return output


def uERF_non_bt(input_tensor, kernel_size, train, summary=False, data_format="NCHW", dropout=0.3, bn_decay=0.95):
  """Builds a NEW full asymmetric conv layer non-bt, with variables and relu.
  Args:
    input_tensor: input tensor
    kernel_size: Size of the kernel [symetric]
    train: If we want to train this layer or not
    data_format: Self explanatory
  Returns:
    output: Output tensor from the convolution
  """
  if data_format == "NCHW":
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[1]
  else:
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[3]
  kernel_nr = prev_depth

  # batchnorm once
  with tf.variable_scope('non_bt'):
    with tf.variable_scope('batchnorm'):
      normalized = tf.contrib.layers.batch_norm(input_tensor,
                                                center=True, scale=True,
                                                is_training=train,
                                                data_format=data_format,
                                                fused=True,
                                                decay=bn_decay,
                                                updates_collections=None)
      if summary:
        variable_summaries(normalized)

    # normal assym bottleneck with relus, no batchnorm
    with tf.variable_scope('asym1'):
      asym1 = asym_conv_layer(normalized, kernel_nr, kernel_size,
                              train, summary=summary, bnorm=False,
                              relu=True, data_format=data_format)

    with tf.variable_scope('asym2'):
      asym2 = asym_conv_layer(asym1, kernel_nr, kernel_size,
                              train, summary=summary, bnorm=False,
                              relu=True, data_format=data_format)

    # add the residual
    with tf.variable_scope('res'):
      res = input_tensor + asym2
      output = tf.layers.dropout(res, rate=dropout, training=train)

  return output


def uERF_downsample(input_tensor, kernel_nr, kernel_size, train,
                    summary=False, data_format="NCHW"):
  """Builds a NEW downsample module, with variables and relu.
  Args:
    input_tensor: input tensor
    kernel_nr: This layer's number of real filters
    kernel_size: Size of the kernel [symmetric]
    train: If we want to train this layer or not
    data_format: Self explanatory
  Returns:
    output: Output tensor from the convolution
  """
  if data_format == "NCHW":
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[1]
  else:
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[3]

  conv_kernels = kernel_nr - prev_depth
  if conv_kernels <= 0:
    print("Wrong number of kernels. Exiting...")
    quit()

  with tf.variable_scope('downsample'):
    with tf.variable_scope('conv'):
      conv = conv_layer(input_tensor, conv_kernels, [kernel_size, kernel_size],
                        2, train, summary=summary, bnorm=True,
                        relu=True, data_format=data_format)

    with tf.variable_scope('pool'):
      pool = max_pool(input_tensor, data_format=data_format)

    with tf.variable_scope('concat'):
      if data_format == "NCHW":
        output = tf.concat([conv, pool], 1)
      else:
        output = tf.concat([conv, pool], 3)

  return output

# definition of pre-softmax layer + its variables (softmax is done by the cost
# function, so to use this model, softmax needs to be applied afterwards)


def linear_layer(input_tensor, classes, train,
                 summary=False, rf=1, data_format="NCHW"):
  """Builds a logit layer that we apply the softmax to, with variables
  Args:
    input_tensor: input tensor
    classes: Number of classes to classify in the output
    train: If we want to train this layer or not
    rf = receptive field of output neuron (for eliminating wrong borders)
    data_format: Self explanatory
  Returns:
    output: Output tensor from the linear layer (end of inference)
  """
  # get previous depth from input tensor
  if data_format == "NCHW":
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[1]
  else:
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[3]

  with tf.variable_scope('weights'):
    W = weight_variable([rf, rf, prev_depth, classes], train)
    if summary:
      variable_summaries(W)

  with tf.variable_scope('biases'):
    b = bias_variable([classes], train)
    if summary:
      variable_summaries(b)

  with tf.variable_scope('linear'):
    output = tf.nn.bias_add(
        conv2d(input_tensor, W, data_format=data_format), b, data_format=data_format)
    if summary:
      variable_summaries(output)

  return output
