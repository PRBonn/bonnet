#!/usr/bin/python3

# Copyright 2017 Andres Milioto, Cyrill Stachniss. All Rights Reserved.
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
import numpy as np


def weight_variable(shape, train):
  print("W: ", shape, "Train:", train)
  return tf.get_variable("w", shape=shape,
                         initializer=tf.variance_scaling_initializer,
                         trainable=train)


def bias_variable(shape, train):
  print("b: ", shape, "Train:", train)
  init = tf.constant(np.full(shape, fill_value=0.1), dtype=tf.float32)
  return tf.get_variable("b",
                         initializer=init,
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


def max_pool(x, k_size=2, stride=2, data_format="NCHW", pad='SAME'):
  # default to a stride of 2 because it is the one we use the most
  if data_format == "NCHW":
    output = tf.nn.max_pool(x,
                            ksize=[1, 1, k_size, k_size],
                            strides=[1, 1, stride, stride],
                            padding=pad, data_format="NCHW")
  else:
    output = tf.nn.max_pool(x,
                            ksize=[1, k_size, k_size, 1],
                            strides=[1, stride, stride, 1],
                            padding=pad, data_format="NHWC")
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


def spatial_dropout(x, keep_prob, training, data_format="NCHW"):
  """
    Drop random channels, using tf.nn.dropout
    (Partially from https://stats.stackexchange.com/questions/282282/how-is-spatial-dropout-in-2d-implemented)
  """
  if training:
    with tf.variable_scope("spatial_dropout"):
      batch_size = x.get_shape().as_list()[0]
      if data_format == "NCHW":
        # depth of previous layer feature map
        prev_depth = x.get_shape().as_list()[1]
        num_feature_maps = [batch_size, prev_depth]
      else:
        # depth of previous layer feature map
        prev_depth = x.get_shape().as_list()[3]
        num_feature_maps = [batch_size, prev_depth]

      # get some uniform noise between keep_prob and 1 + keep_prob
      random_tensor = keep_prob
      random_tensor += tf.random_uniform(num_feature_maps,
                                         dtype=x.dtype)

      # if we take the floor of this, we get a binary matrix where
      # (1-keep_prob)% of the values are 0 and the rest are 1
      binary_tensor = tf.floor(random_tensor)

      # Reshape to multiply our feature maps by this tensor correctly
      if data_format == "NCHW":
        binary_tensor = tf.reshape(binary_tensor,
                                   [batch_size, prev_depth, 1, 1])
      else:
        binary_tensor = tf.reshape(binary_tensor,
                                   [batch_size, 1, 1, prev_depth])

      # Zero out feature maps where appropriate; scale up to compensate
      ret = tf.div(x, keep_prob) * binary_tensor
  else:
    ret = x

  return ret


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
    preactivations = conv2d(input_tensor, W, stride,
                            data_format=data_format)
    if summary:
      variable_summaries(preactivations)

  if bnorm:
    with tf.variable_scope('batchnorm'):
      normalized = tf.contrib.layers.batch_norm(preactivations,
                                                center=True, scale=True,
                                                is_training=train,
                                                data_format=data_format,
                                                fused=True,
                                                decay=bn_decay)
      if summary:
        variable_summaries(normalized)
  else:
    normalized = preactivations

  if relu:
    with tf.variable_scope('relu'):
      output = relu = tf.nn.leaky_relu(normalized)
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
  print("W: ", [2, 2, prev_depth, kernel_nr], "Train:", train)

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
                      summary=summary, bnorm=bnorm, relu=False, data_format=data_format)
  with tf.variable_scope('vert'):
    output = conv_layer(conv, kernel_nr, [1, kernel_size], 1, train,
                        summary=summary, bnorm=bnorm, relu=relu, data_format=data_format)

  return output


def inception(input_tensor, train, summary=False, data_format="NCHW", dropout=0.3, bn_decay=0.95):
  """Builds a NEW full asymmetric conv layer non-bt, with variables and relu.
  Args:
    input_tensor: input tensor
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

  if prev_depth % 4:
    print("Warning! Depth cannot be divided by 4 in inception module")

  # kernel number to match input
  kernel_nr = int(prev_depth / 4)

  with tf.variable_scope('inception'):
    with tf.variable_scope('bottleneck'):
      bt = conv_layer(input_tensor, kernel_nr, [1, 1],
                      1, train, summary=summary, bnorm=False,
                      relu=False, data_format=data_format)

    with tf.variable_scope('3x3'):
      asym3 = asym_conv_layer(bt, kernel_nr * 2, 3,
                              train, summary=summary, bnorm=False,
                              relu=False, data_format=data_format)

    with tf.variable_scope('5x5'):
      asym5 = asym_conv_layer(bt, kernel_nr, 5,
                              train, summary=summary, bnorm=False,
                              relu=False, data_format=data_format)

    with tf.variable_scope('7x7'):
      asym7 = asym_conv_layer(bt, kernel_nr, 7,
                              train, summary=summary, bnorm=False,
                              relu=False, data_format=data_format)

    with tf.variable_scope('concat'):
      if data_format == "NCHW":
        concat = tf.concat([asym3, asym5, asym7], 1)
      else:
        concat = tf.concat([asym3, asym5, asym7], 3)

    with tf.variable_scope('batchnorm'):
      # use batch renorm for small minibatches
      normalized = tf.contrib.layers.batch_norm(concat,
                                                center=True, scale=True,
                                                is_training=train,
                                                data_format=data_format,
                                                fused=True,
                                                decay=bn_decay,
                                                renorm=True)
      if summary:
        variable_summaries(normalized)

    # add the residual
    with tf.variable_scope('out'):
      drop = spatial_dropout(normalized, keep_prob=1 - dropout,
                             training=train, data_format=data_format)
      output = tf.nn.leaky_relu(drop)

  return output


def dense_inception(input_tensor, n_blocks, train, summary=False, data_format="NCHW", dropout=0.3, bn_decay=0.95):
  """Builds a NEW full asymmetric conv layer non-bt, with variables and relu.
  Args:
    input_tensor: input tensor
    n_blocks: number of layers inside block
    train: If we want to train this layer or not
    data_format: Self explanatory
  Returns:
    output: Output tensor from the convolution
  """
  with tf.variable_scope('dense_inception'):
    if data_format == "NCHW":
      # depth of previous layer feature map
      prev_depth = input_tensor.get_shape().as_list()[1]
    else:
      # depth of previous layer feature map
      prev_depth = input_tensor.get_shape().as_list()[3]

    blocks = input_tensor
    for b in range(n_blocks):
      with tf.variable_scope('skip_conv_' + str(b)):
        out_block = inception(blocks, train, summary=summary,
                              data_format=data_format, dropout=dropout,
                              bn_decay=bn_decay)
        if data_format == "NCHW":
          blocks = tf.concat([blocks, out_block], 1)
        else:
          blocks = tf.concat([blocks, out_block], 3)

    # linear squash
    with tf.variable_scope('squash'):
      squash = conv_layer(blocks, prev_depth, [1, 1], 1, train,
                          summary=summary, bnorm=False, relu=False,
                          data_format=data_format)

    with tf.variable_scope('res'):
      output = squash + input_tensor

  return output


def inv_residual(input_tensor, channel_mult, train, summary=False,
                 data_format="NCHW", dropout=0.3, bn_decay=0.95):
  """Builds a NEW full asymmetric conv layer non-bt, with variables and relu.
  Args:
    input_tensor: input tensor
    channel_mult: number of filters in each inverted residual (ratio with input filters)
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

  # number filters
  n_filters = channel_mult * prev_depth

  with tf.variable_scope("inverted_residual"):
    with tf.variable_scope("conv"):
      with tf.variable_scope("bnorm"):
        conv_norm = tf.contrib.layers.batch_norm(input_tensor,
                                                 center=True, scale=True,
                                                 is_training=train,
                                                 data_format=data_format,
                                                 fused=True,
                                                 decay=bn_decay)

      with tf.variable_scope("conv"):
        with tf.variable_scope("depthwise_filter"):
          depthwise_filter = weight_variable([3, 3, prev_depth, channel_mult], train)
          if summary:
            variable_summaries(depthwise_filter)
        with tf.variable_scope("pointwise_filter"):
          pointwise_filter = weight_variable(
              [1, 1, n_filters, prev_depth], train)
          if summary:
            variable_summaries(pointwise_filter)
        with tf.variable_scope("conv"):
          if data_format == "NCHW":
            conv = tf.nn.separable_conv2d(conv_norm,
                                          depthwise_filter,
                                          pointwise_filter,
                                          strides=[1, 1, 1, 1],
                                          padding="SAME",
                                          data_format="NCHW")
          else:
            conv = tf.nn.separable_conv2d(conv_norm,
                                          depthwise_filter,
                                          pointwise_filter,
                                          strides=[1, 1, 1, 1],
                                          padding="SAME",
                                          data_format="NHWC")

      with tf.variable_scope("residual"):
        dropout = spatial_dropout(conv, keep_prob=1 - dropout,
                                  training=train, data_format=data_format)

        residual = dropout + input_tensor

      with tf.variable_scope("out"):
        output = tf.nn.leaky_relu(residual)

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
    # normal assym bottleneck with relus, no batchnorm
    with tf.variable_scope('asym1'):
      asym1 = asym_conv_layer(input_tensor, kernel_nr, kernel_size,
                              train, summary=summary, bnorm=False,
                              relu=True, data_format=data_format)

    with tf.variable_scope('asym2'):
      asym2 = asym_conv_layer(asym1, kernel_nr, kernel_size,
                              train, summary=summary, bnorm=False,
                              relu=False, data_format=data_format)

    with tf.variable_scope('batchnorm'):
      normalized = tf.contrib.layers.batch_norm(asym2,
                                                center=True, scale=True,
                                                is_training=train,
                                                data_format=data_format,
                                                fused=True,
                                                decay=bn_decay)
      if summary:
        variable_summaries(normalized)

    # add the residual
    with tf.variable_scope('res'):
      drop = spatial_dropout(normalized, keep_prob=1 - dropout,
                             training=train, data_format=data_format)
      with tf.variable_scope('relu'):
        relu = tf.nn.leaky_relu(input_tensor + drop)
      output = relu

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


def psp_layer(input_tensor, piramids, biggest_piramid, train, summary=False, data_format="NCHW"):
  """Builds a psp layer to get context info.
  Args:
    input_tensor: input tensor
    piramids: number of pooling layers to use
    biggest_piramid: Size of biggest pooling kernel
    train: If we want to train this layer or not
    data_format: Self explanatory
  Returns:
    output: Output tensor from the convolution
  """
  if data_format == "NCHW":
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[1]
    prev_h = input_tensor.get_shape().as_list()[2]
    prev_w = input_tensor.get_shape().as_list()[3]
  else:
    # depth of previous layer feature map
    prev_depth = input_tensor.get_shape().as_list()[3]
    prev_h = input_tensor.get_shape().as_list()[1]
    prev_w = input_tensor.get_shape().as_list()[2]
  batch_size = input_tensor.get_shape().as_list()[0]

  # calculate pooling kernel sizes
  min_h_w = min(prev_h, prev_w)
  if biggest_piramid > min_h_w:
    print("WARNING! Biggest piramid is bigger than the shortest code dimension")
  kernel_sizes = []
  for i in range(piramids):
    k_size = int(biggest_piramid / (2 ** i))
    kernel_sizes.append(k_size)
    print("PSP KSIZE: ", k_size)

  # number of convolutions for each level
  nodes = input_tensor
  n_convs = int(prev_depth / float(len(kernel_sizes)))
  print("NCONVS: ", n_convs)

  with tf.variable_scope('psp'):
    p = piramids
    for k_size in kernel_sizes:
      print('psp-' + str(p))
      with tf.variable_scope('psp-' + str(p)):
        p -= 1  # if I call the module psp-ksize it will break when retraining
        # pool
        pool = max_pool(input_tensor, k_size=k_size,
                        stride=1, data_format=data_format, pad='VALID')
        # conv
        conv = conv_layer(pool, n_convs, (5, 5), 1, train, summary=summary,
                          bnorm=False, relu=True, data_format=data_format)

        # upsample kernel and deconv
        # make kernel with 1's only where I want to upsample etc,etc
        k_shape = (k_size, k_size, n_convs, n_convs)
        k = np.zeros(k_shape)
        ones = np.ones((k_size, k_size))
        for i in range(n_convs):
          k[:, :, i, i] = ones
        k_tf = tf.constant(k, dtype=tf.float32)
        print("Upsample k: ", k_shape)
        if data_format == "NCHW":
          upsample = tf.nn.conv2d_transpose(conv,
                                            k_tf,
                                            output_shape=(batch_size, n_convs,
                                                          prev_h, prev_w),
                                            strides=(1, 1, 1, 1),
                                            padding='VALID',
                                            data_format=data_format)
          nodes = tf.concat([nodes, upsample], 1)
        else:
          upsample = tf.nn.conv2d_transpose(conv,
                                            k_tf,
                                            output_shape=(batch_size, prev_h,
                                                          prev_w, n_convs),
                                            strides=(1, 1, 1, 1),
                                            padding='VALID',
                                            data_format=data_format)
          nodes = tf.concat([nodes, upsample], 3)

  return nodes


def reduce(input_tensor, skip, n_kernels, train, summary=False, data_format="NCHW"):
  """ Concatenate input tensor and skip connection and apply a 1x1 conv
  to reduce to n_kernels depth.
  Args:
    input_tensor: input tensor
    skip: skip connection from encoder
    n_kernels: number of new kernels
    train: If we want to train this layer or not
    summary: Save summaries
    data_format: Self explanatory
  Returns:
    output: Output tensor from the op
  """
  with tf.variable_scope("reduce"):
    if data_format == "NCHW":
      concat = tf.concat([input_tensor, skip], 1)
    else:
      concat = tf.concat([input_tensor, skip], 3)

    output = conv_layer(concat, n_kernels, [1, 1], 1, train,
                        summary=summary, bnorm=True, relu=True, data_format=data_format)

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
