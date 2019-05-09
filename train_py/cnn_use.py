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
  Use the trained network on an input image.
'''
import os
import argparse
import imp
import yaml
import time

# image plot stuff
import cv2
import numpy as np
import dataset.aux_scripts.util as util
import scipy.io as sio

# tensorflow stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up TF!
import tensorflow as tf
import signal


def predict_mask(img, net, FLAGS, DATA):
  # open image
  cvim = cv2.imread(img, cv2.IMREAD_UNCHANGED)
  if cvim is None:
    print("No image to open for ", img)
    return
  # predict mask from image
  start = time.time()
  mask = net.predict(cvim, path=FLAGS.path + '/' +
                     FLAGS.model, verbose=FLAGS.verbose)
  print("Prediction for img ", img, ". Elapsed: ", time.time() - start, "s")
  # change to color
  color_mask = util.prediction_to_color(
      mask, DATA["label_remap"], DATA["color_map"])

  # assess accuracy (if wanted)
  if FLAGS.label is not None:
    label = cv2.imread(FLAGS.label, 0)
    if label is None:
      print("No label to open")
      quit()
    net.individual_accuracy(mask, label)

  cv2.imwrite(FLAGS.log + "/" + os.path.basename(img), color_mask)

  if FLAGS.verbose:
    # show me the image
    # first, mix with image
    im, transparent_mask = util.transparency(cvim, color_mask)
    all_img = np.concatenate((im, transparent_mask, color_mask), axis=1)
    util.im_tight_plt(all_img)
    util.im_block()

  return


def predict_probs(img, net, FLAGS, DATA):
  # open image
  cvim = cv2.imread(img, cv2.IMREAD_UNCHANGED)
  if cvim is None:
    print("No image to open for ", img)
    return
  # predict mask from image
  start = time.time()
  probs = net.predict(cvim, path=FLAGS.path + '/' +
                      FLAGS.model, verbose=FLAGS.verbose, as_probs=True)
  print("Prediction for img ", img, ". Elapsed: ", time.time() - start, "s")

  # save to matlab matrix
  matname = FLAGS.log + "/" + \
      os.path.splitext(os.path.basename(img))[0] + ".mat"
  sio.savemat(matname, {'p': probs})

  return


def predict_code(img, net, FLAGS):
  # predict feature map from image
  # open image
  cvim = cv2.imread(img, cv2.IMREAD_UNCHANGED)
  if cvim is None:
    print("No image to open for ", img)
    return
  # predict mask from image
  start = time.time()
  code = net.predict_code(cvim, path=FLAGS.path + '/' +
                          FLAGS.model, verbose=FLAGS.verbose)
  print("Prediction for img ", img, ". Elapsed: ", time.time() - start, "s")

  # reshape code to single dimension
  reshaped_code = np.reshape(code, (1, -1))
  # print("Shape", reshaped_code.shape)

  # save code to text file
  filename = FLAGS.log + "/" + \
      os.path.splitext(os.path.basename(img))[0] + ".txt"
  print("Saving feature map to: ", filename)
  np.savetxt(filename, reshaped_code, fmt="%.8f", delimiter=" ")

  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cnn_use.py")
  parser.add_argument(
      '--image', '-i',
      nargs='+',
      type=str,
      required=True,
      help='Image to infer. No Default',
  )
  parser.add_argument(
      '--label', '--lbl',
      type=str,
      required=False,
      default=None,
      help='Label to assess accuracy, if wanted. Only works for the first image',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default='/tmp/net_predict_log',
      help='Directory to log output of predictions. Defaults to %(default)s',
  )
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=True,
      help='Directory to get the model. No default!'
  )
  model_choices = ['acc', 'iou']
  parser.add_argument(
      '--model', '-m',
      type=str,
      default='iou',
      help='Type of model (best acc or best iou). Default to %(default)s',
      choices=model_choices
  )
  parser.add_argument(
      '--data', '-d',
      type=str,
      help='Dataset yaml cfg file. See /cfg for sample. Defaults to the one in log dir',
  )
  parser.add_argument(
      '--net', '-n',
      type=str,
      help='Network yaml cfg file. See /cfg for sample. Defaults to the one in log dir',
  )
  parser.add_argument(
      '--train', '-t',
      type=str,
      help='Training hyperparameters yaml cfg file. Defaults to the one in log dir',
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
  parser.add_argument(
      '--probs',
      dest='probs',
      default=False,
      action='store_true',
      help='Probability mode. Calculates probability map instead of mask. Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Image to infer: ", FLAGS.image)
  print("Label: ", FLAGS.label)
  print("Log dir: ", FLAGS.log)
  print("model path", FLAGS.path)
  print("model type", FLAGS.model)
  print("data yaml: ", FLAGS.data)
  print("net yaml: ", FLAGS.net)
  print("train yaml: ", FLAGS.train)
  print("Verbose?: ", FLAGS.verbose)
  print("Features?: ", FLAGS.code)
  print("Probabilities?: ", FLAGS.probs)
  print("----------\n")

  # try to open data yaml
  try:
    if(FLAGS.data):
      print("Opening desired data file %s" % FLAGS.data)
      f = open(FLAGS.data, 'r')
    else:
      print("Opening default data file data.yaml from log folder")
      f = open(FLAGS.path + '/data.yaml', 'r')
    DATA = yaml.load(f)
  except:
    print("Error opening data yaml file...")
    quit()

  # try to open net yaml
  try:
    if(FLAGS.net):
      print("Opening desired net file %s" % FLAGS.net)
      f = open(FLAGS.net, 'r')
    else:
      print("Opening default net file net.yaml from log folder")
      f = open(FLAGS.path + '/net.yaml', 'r')
    NET = yaml.load(f)
  except:
    print("Error opening net yaml file...")
    quit()

  # try to open train yaml
  try:
    if(FLAGS.train):
      print("Opening desired train file %s" % FLAGS.train)
      f = open(FLAGS.train, 'r')
    else:
      print("Opening default train file train.yaml from log folder")
      f = open(FLAGS.path + '/train.yaml', 'r')
    TRAIN = yaml.load(f)
  except:
    print("Error opening train yaml file...")
    quit()

  # create log folder
  if tf.gfile.Exists(FLAGS.path + '/' + FLAGS.model):
    print("Model folder exists! Using model from %s" %
          (FLAGS.path + '/' + FLAGS.model))

  # get architecture
  architecture = imp.load_source("architecture",
                                 os.getcwd() + '/arch/' +
                                 NET["name"] + '.py')

  # build the network
  net = architecture.Network(DATA, NET, TRAIN, FLAGS.log)

  # handle ctrl-c for threads
  signal.signal(signal.SIGINT, net.cleanup)
  signal.signal(signal.SIGTERM, net.cleanup)
  # signal.pause()

  try:
    if tf.gfile.Exists(FLAGS.log):
      tf.gfile.DeleteRecursively(FLAGS.log)
    tf.gfile.MakeDirs(FLAGS.log)
  except:
    print("Error creating log directory. Check permissions! Exiting...")
    quit()

  if type(FLAGS.image) is not list:
    images = [FLAGS.image]
  else:
    images = FLAGS.image

  for img in images:
    # predict
    if FLAGS.code:
      predict_code(img, net, FLAGS)
    elif FLAGS.probs:
      predict_probs(img, net, FLAGS, DATA)
    else:
      predict_mask(img, net, FLAGS, DATA)

  # clean up
  net.cleanup(None, None)
