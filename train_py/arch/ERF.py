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
    if len(train_lyr) != 25:
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

      with tf.variable_scope("Block1"):
        print("Block 1")
        block1_lyr1 = lyr.uERF_downsample(n_img, n_k_lyr[0], 3,
                                          train_stage and train_lyr[0],
                                          summary,
                                          data_format=data_format)

        block_1 = block1_lyr1

      with tf.variable_scope("Block2"):
        print("Block 2")
        block2_lyr1 = lyr.uERF_downsample(block_1, n_k_lyr[1], 3,
                                          train_stage and train_lyr[1],
                                          summary,
                                          data_format=data_format)

        with tf.variable_scope("non-bt-1"):
          block2_lyr2 = lyr.uERF_non_bt(block2_lyr1, 3,
                                        train_stage and train_lyr[2],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-2"):
          block2_lyr3 = lyr.uERF_non_bt(block2_lyr2, 3,
                                        train_stage and train_lyr[3],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-3"):
          block2_lyr4 = lyr.uERF_non_bt(block2_lyr3, 3,
                                        train_stage and train_lyr[4],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-4"):
          block2_lyr5 = lyr.uERF_non_bt(block2_lyr4, 3,
                                        train_stage and train_lyr[5],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        block_2 = block2_lyr5

      with tf.variable_scope("Block3"):
        print("Block 3")
        block3_lyr1 = lyr.uERF_downsample(block_2, n_k_lyr[2], 3,
                                          train_stage and train_lyr[6],
                                          summary,
                                          data_format=data_format)

        with tf.variable_scope("non-bt-1"):
          block3_lyr2 = lyr.uERF_non_bt(block3_lyr1, 3,
                                        train_stage and train_lyr[7],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-2"):
          block3_lyr3 = lyr.uERF_non_bt(block3_lyr2, 5,
                                        train_stage and train_lyr[8],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-3"):
          block3_lyr4 = lyr.uERF_non_bt(block3_lyr3, 7,
                                        train_stage and train_lyr[9],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-4"):
          block3_lyr5 = lyr.uERF_non_bt(block3_lyr4, 9,
                                        train_stage and train_lyr[10],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-5"):
          block3_lyr6 = lyr.uERF_non_bt(block3_lyr5, 3,
                                        train_stage and train_lyr[11],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-6"):
          block3_lyr7 = lyr.uERF_non_bt(block3_lyr6, 5,
                                        train_stage and train_lyr[12],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-7"):
          block3_lyr8 = lyr.uERF_non_bt(block3_lyr7, 7,
                                        train_stage and train_lyr[13],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-8"):
          block3_lyr9 = lyr.uERF_non_bt(block3_lyr8, 9,
                                        train_stage and train_lyr[14],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        code = block_3 = block3_lyr9

    # end encoder, start decoder
    print("============= End of encoder ===============")
    print("size of code: ", code.get_shape().as_list())
    print("=========== Beginning of decoder============")

    with tf.variable_scope("decoder"):
      print("decoder")

      with tf.variable_scope("Block4"):
        print("Block 4")
        with tf.variable_scope("unpool"):
          block4_unpool = lyr.upsample_layer(code,
                                             train_stage and train_lyr[15],
                                             kernels=n_k_lyr[3],
                                             data_format=data_format)

        with tf.variable_scope("non-bt-1"):
          block4_lyr1 = lyr.uERF_non_bt(block4_unpool, 3,
                                        train_stage and train_lyr[16],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-2"):
          block4_lyr2 = lyr.uERF_non_bt(block4_lyr1, 3,
                                        train_stage and train_lyr[17],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        block_4 = block4_lyr2

      with tf.variable_scope("Block5"):
        print("Block 5")
        with tf.variable_scope("unpool"):
          block5_unpool = lyr.upsample_layer(block_4,
                                             train_stage and train_lyr[18],
                                             kernels=n_k_lyr[4],
                                             data_format=data_format)

        with tf.variable_scope("non-bt-1"):
          block5_lyr1 = lyr.uERF_non_bt(block5_unpool, 3,
                                        train_stage and train_lyr[19],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        with tf.variable_scope("non-bt-2"):
          block5_lyr2 = lyr.uERF_non_bt(block5_lyr1, 3,
                                        train_stage and train_lyr[20],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

        block_5 = block5_lyr2

      with tf.variable_scope("Block6"):
        print("Block 6")
        with tf.variable_scope("unpool"):
          block6_unpool = lyr.upsample_layer(block_5,
                                             train_stage and train_lyr[21],
                                             kernels=n_k_lyr[5],
                                             data_format=data_format)

        with tf.variable_scope("non-bt-1"):
          block6_lyr1 = lyr.uERF_non_bt(block6_unpool, 3,
                                        train_stage and train_lyr[22],
                                        summary,
                                        data_format=data_format,
                                        dropout=self.NET["dropout"],
                                        bn_decay=self.NET["bn_decay"])

          block_6 = block6_lyr1

    with tf.variable_scope("logits"):
      # convert to logits with a linear layer
      logits_linear = lyr.linear_layer(block_6, self.num_classes,
                                       train_stage and train_lyr[24],
                                       summary=summary,
                                       data_format=data_format)

    # transpose logits back to NHWC
    if data_format == "NCHW":
      logits = tf.transpose(logits_linear, [0, 2, 3, 1])
    else:
      logits = logits_linear

    return logits, code, n_img
