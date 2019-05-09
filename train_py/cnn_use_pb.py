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
  Use the trained frozen pb on an input image.
'''
import os
import argparse
import yaml
import time

# image plot stuff
import cv2
import numpy as np

import dataset.aux_scripts.util as util

# tensorflow stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up TF!
import tensorflow as tf
import signal


def predict_mask(img, sess, input, output, FLAGS, DATA):
  # open image
  cvim = cv2.imread(img, cv2.IMREAD_UNCHANGED)
  if cvim is None:
    print("No image to open for ", img)
    return
  # predict mask from image
  start = time.time()
  mask = sess.run(output, feed_dict={input: [cvim]})
  print("Prediction for img ", img, ". Elapsed: ", time.time() - start, "s")
  # change to color
  color_mask = util.prediction_to_color(
      mask[0, :, :], DATA["label_remap"], DATA["color_map"])

  cv2.imwrite(FLAGS.log + "/" + os.path.basename(img), color_mask)

  if FLAGS.verbose:
    # show me the image
    # first, mix with image
    im, transparent_mask = util.transparency(cvim, color_mask)
    all_img = np.concatenate((im, transparent_mask, color_mask), axis=1)
    util.im_tight_plt(all_img)
    util.im_block()

  return


def predict_code(img, sess, input, output, FLAGS):
  # predict feature map from image
  # open image
  cvim = cv2.imread(img, cv2.IMREAD_UNCHANGED)
  if cvim is None:
    print("No image to open for ", img)
    return

  # predict code from image
  print("Prediction for img ", img)
  code = sess.run(output, feed_dict={input: [cvim]})

  # reshape code to single dimension
  reshaped_code = np.reshape(code, (1, -1))
  print("Shape", reshaped_code.shape)

  # save code to text file
  filename = FLAGS.log + "/" + \
      os.path.splitext(os.path.basename(img))[0] + ".txt"
  print("Saving feature map to: ", filename)
  np.savetxt(filename, reshaped_code, fmt="%.8f", delimiter=" ")

  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cnn_use_pb.py")
  parser.add_argument(
      '--image', '-i',
      nargs='+',
      type=str,
      required=True,
      help='Image to infer. No Default',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default='/tmp/pb_predictions/',
      help='Directory to log output of predictions. Defaults to %(default)s',
  )
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=True,
      help='Directory to get the model. No default!'
  )
  model_choices = ['frozen_nchw', 'frozen_nhwc', 'optimized', 'quantized']
  parser.add_argument(
      '--model', '-m',
      type=str,
      default='frozen_nchw',
      help='Type of model (frozen or optimized). Default to %(default)s',
      choices=model_choices
  )
  parser.add_argument(
      '--verbose', '-v',
      dest='verbose',
      default=False,
      action='store_true',
      help='Verbose mode. Calculates profile. Defaults to %(default)s',
  )
  parser.add_argument(
      '--code', '-c',
      dest='code',
      default=False,
      action='store_true',
      help='Code mode. Calculates feature map instead of mask. Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Image to infer: ", FLAGS.image)
  print("Log dir: ", FLAGS.log)
  print("model path", FLAGS.path)
  print("model type", FLAGS.model)
  print("Verbose?: ", FLAGS.verbose)
  print("Features?: ", FLAGS.code)
  print("----------\n")

  # try to open data yaml
  try:
    print("Opening default data file data.yaml from log folder")
    f = open(FLAGS.path + '/data.yaml', 'r')
    DATA = yaml.load(f)
  except:
    print("Error opening data yaml file...")
    quit()

  # try to open net yaml
  try:
    print("Opening default net file net.yaml from log folder")
    f = open(FLAGS.path + '/net.yaml', 'r')
    NET = yaml.load(f)
  except:
    print("Error opening net yaml file...")
    quit()

  # try to open train yaml
  try:
    print("Opening default train file train.yaml from log folder")
    f = open(FLAGS.path + '/train.yaml', 'r')
    TRAIN = yaml.load(f)
  except:
    print("Error opening train yaml file...")
    quit()

  # try to open nodes yaml
  try:
    print("Opening default nodes file nodes.yaml from log folder")
    f = open(FLAGS.path + '/nodes.yaml', 'r')
    NODES = yaml.load(f)
  except:
    print("Error opening nodes yaml file...")
    quit()

  frozen_name = os.path.join(FLAGS.path, FLAGS.model + ".pb")
  if tf.gfile.Exists(frozen_name):
    print("Model file exists! Using model from %s" % (frozen_name))
  else:
    print("Model not found. Exiting...")
    quit()

  # create log folder
  try:
    if tf.gfile.Exists(FLAGS.log):
      tf.gfile.DeleteRecursively(FLAGS.log)
    tf.gfile.MakeDirs(FLAGS.log)
  except:
    print("Error creating log directory. Check permissions! Exiting...")
    quit()

  # node names
  input_node = NODES["input_node"] + ':0'
  code_node = NODES["code_node"] + ':0'
  mask_node = NODES["mask_node"] + ':0'

  with tf.Graph().as_default() as graph:
    # open graph def from frozen model
    try:
      with tf.gfile.GFile(frozen_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    except:
      print("Failed to extract grapfdef. Exiting...")
      quit()

    # import the graph
    pl, code, mask = tf.import_graph_def(graph_def, return_elements=[
                                         input_node, code_node, mask_node])

    # infer from pb
    gpu_options = tf.GPUOptions(allow_growth=True, force_gpu_compatible=True)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False, gpu_options=gpu_options)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2

    # start a session
    sess = tf.Session(config=config)

    # process images
    if type(FLAGS.image) is not list:
      images = [FLAGS.image]
    else:
      images = FLAGS.image

    # use model for prediction
    for img in images:
      # predict
      if FLAGS.code:
        predict_code(img, sess, pl, code, FLAGS)
      else:
        predict_mask(img, sess, pl, mask, FLAGS, DATA)
