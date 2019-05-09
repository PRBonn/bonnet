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
  Network class, containing definition of the graph
  API Style should be the same for all nets (Same class name and member functions)
'''
# tf
import tensorflow as tf

# common layers
from arch.abstract_net import AbstractNetwork
import arch.layer as lyr


class Network(AbstractNetwork):
  def __init__(self, DATA, NET, TRAIN, logdir):
    # init parent
    super().__init__(DATA, NET, TRAIN, logdir)

  def build_graph(self, img_pl, train_stage, data_format="NCHW"):
    # some graph info depending on what I will do with it
    summary = self.TRAIN['summary']
    train_lyr = self.NET['train_lyr']
    n_k_lyr = self.NET['n_k_lyr']
    n_lyr = self.NET['n_lyr']
    if len(train_lyr) != 7:
      print("Wrong length in train list for network. Exiting...")
      quit()
    self.num_classes = len(self.DATA['label_map'])

    # build the graph
    print("Building graph")

    with tf.variable_scope('images'):
      # resize input to desired size
      img_resized = tf.image.resize_images(img_pl,
                                           [self.DATA["img_prop"]["height"],
                                            self.DATA["img_prop"]["width"]])
      # if on GPU. transpose to NCHW
      if data_format == "NCHW":
        # convert from NHWC to NCHW (faster on GPU)
        img_transposed = tf.transpose(img_resized, [0, 3, 1, 2])
      else:
        img_transposed = img_resized
      # normalization of input
      n_img = (img_transposed - 128) / 128

    with tf.variable_scope("encoder"):
      print("encoder")
      with tf.variable_scope("downsample1"):
        print("downsample1")
        # input image 1024*512 - 960*720
        down_lyr1 = lyr.uERF_downsample(n_img, n_k_lyr[0], 3,
                                        train_stage and train_lyr[0],
                                        summary,
                                        data_format=data_format)

        inv_residual_lyr1 = [down_lyr1]
        for n in range(n_lyr[0]):
          with tf.variable_scope("inv-res-" + str(n)):
            inv_residual_lyr1.append(lyr.inv_residual(inv_residual_lyr1[-1], n_k_lyr[1],
                                                      train_stage and train_lyr[0],
                                                      summary,
                                                      data_format=data_format,
                                                      dropout=self.NET["dropout"],
                                                      bn_decay=self.NET["bn_decay"]))

          downsample1 = inv_residual_lyr1[-1]

      with tf.variable_scope("downsample2"):
        print("downsample2")
        # input image 512*256 - 480*360
        down_lyr2 = lyr.uERF_downsample(downsample1, n_k_lyr[2], 3,
                                        train_stage and train_lyr[1],
                                        summary,
                                        data_format=data_format)

        inv_residual_lyr2 = [down_lyr2]
        for n in range(n_lyr[1]):
          with tf.variable_scope("inv-res-" + str(n)):
            inv_residual_lyr2.append(lyr.inv_residual(inv_residual_lyr2[-1], n_k_lyr[3],
                                                      train_stage and train_lyr[1],
                                                      summary,
                                                      data_format=data_format,
                                                      dropout=self.NET["dropout"],
                                                      bn_decay=self.NET["bn_decay"]))

        downsample2 = inv_residual_lyr2[-1]

      with tf.variable_scope("downsample3"):
        print("downsample3")
        # input image 512*256 - 480*360
        down_lyr3 = lyr.uERF_downsample(downsample2, n_k_lyr[4], 3,
                                        train_stage and train_lyr[2],
                                        summary,
                                        data_format=data_format)

        inv_residual_lyr3 = [down_lyr3]
        for n in range(n_lyr[2]):
          with tf.variable_scope("inv-res-" + str(n)):
            inv_residual_lyr3.append(lyr.inv_residual(inv_residual_lyr3[-1], n_k_lyr[5],
                                                      train_stage and train_lyr[2],
                                                      summary,
                                                      data_format=data_format,
                                                      dropout=self.NET["dropout"],
                                                      bn_decay=self.NET["bn_decay"]))

        downsample3 = inv_residual_lyr3[-1]

    code = downsample3

    # end encoder, start decoder
    print("============= End of encoder ===============")
    print("size of code: ", code.get_shape().as_list())
    print("=========== Beginning of decoder============")

    with tf.variable_scope("decoder"):
      print("decoder")

      with tf.variable_scope("upsample"):
        print("upsample")
        with tf.variable_scope("unpool1"):
          print("unpool1")
          unpool_lyr1 = lyr.upsample_layer(code,
                                           train_stage and train_lyr[3],
                                           kernels=n_k_lyr[6],
                                           data_format=data_format) + downsample2

          inv_residual_unpool_1 = [unpool_lyr1]
          for n in range(n_lyr[4]):
            with tf.variable_scope("inv-res-" + str(n)):
              inv_residual_unpool_1.append(lyr.inv_residual(inv_residual_unpool_1[-1], n_k_lyr[7],
                                                            train_stage and train_lyr[3],
                                                            summary,
                                                            data_format=data_format,
                                                            dropout=self.NET["dropout"],
                                                            bn_decay=self.NET["bn_decay"]))

          unpool1 = inv_residual_unpool_1[-1]

        with tf.variable_scope("unpool2"):
          print("unpool2")
          unpool_lyr2 = lyr.upsample_layer(unpool1,
                                           train_stage and train_lyr[4],
                                           kernels=n_k_lyr[8],
                                           data_format=data_format) + downsample1

          inv_residual_unpool_2 = [unpool_lyr2]
          for n in range(n_lyr[4]):
            with tf.variable_scope("inv-res-" + str(n)):
              inv_residual_unpool_2.append(lyr.inv_residual(inv_residual_unpool_2[-1], n_k_lyr[9],
                                                            train_stage and train_lyr[4],
                                                            summary,
                                                            data_format=data_format,
                                                            dropout=self.NET["dropout"],
                                                            bn_decay=self.NET["bn_decay"]))

          unpool2 = inv_residual_unpool_2[-1]

        with tf.variable_scope("unpool3"):
          print("unpool3")
          # input image 64*32 - 60*45
          unpool_lyr3 = lyr.upsample_layer(unpool2 + inv_residual_lyr1[-1],
                                           train_stage and train_lyr[5],
                                           kernels=n_k_lyr[10],
                                           data_format=data_format)

          inv_residual_unpool_3 = [unpool_lyr3]
          for n in range(n_lyr[4]):
            with tf.variable_scope("inv-res-" + str(n)):
              inv_residual_unpool_3.append(lyr.inv_residual(inv_residual_unpool_3[-1], n_k_lyr[11],
                                                            train_stage and train_lyr[5],
                                                            summary,
                                                            data_format=data_format,
                                                            dropout=self.NET["dropout"],
                                                            bn_decay=self.NET["bn_decay"]))

          unpool3 = inv_residual_unpool_3[-1]

      unpooled = unpool3

    with tf.variable_scope("logits"):
      # convert to logits with a linear layer
      logits_linear = lyr.linear_layer(unpooled, self.num_classes,
                                       train_stage and train_lyr[6],
                                       summary=summary,
                                       data_format=data_format)

    # transpose logits back to NHWC
    if data_format == "NCHW":
      logits = tf.transpose(logits_linear, [0, 2, 3, 1])
    else:
      logits = logits_linear

    return logits, code, n_img
