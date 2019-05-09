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
  Use the network on an input image, input video, or entire dataset to analyze
  performance.
'''
import os
import argparse
import imp
import yaml

# tensorflow stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up TF!
import tensorflow as tf
import signal

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cnn_dataset_performance.py")
  parser.add_argument(
      '--dataset',
      type=str,
      required=True,
      help='Image to infer. No Default',
  )
  parser.add_argument(
      '--batchsize', '-b',
      type=int,
      required=False,
      default=1,
      help='Image to infer. Defaults to %(default)s',
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
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Dataset: ", FLAGS.dataset)
  print("Batchsize: ", FLAGS.batchsize)
  print("Log dir: ", FLAGS.log)
  print("model path", FLAGS.path)
  print("model type", FLAGS.model)
  print("data yaml: ", FLAGS.data)
  print("net yaml: ", FLAGS.net)
  print("train yaml: ", FLAGS.train)
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

  # try to get model
  if tf.gfile.Exists(FLAGS.path + '/' + FLAGS.model):
    print("Model folder exists! Using model from %s" %
          (FLAGS.path + '/' + FLAGS.model))
  else:
    print("Model does not exist")
    quit()

  # try to get dataset
  if tf.gfile.Exists(FLAGS.dataset):
    print("Dataset folder exists!")
  else:
    print("Model does not exist. Gimme data. Exiting...")
    quit()

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

  # create log dir
  try:
    if tf.gfile.Exists(FLAGS.log):
      tf.gfile.DeleteRecursively(FLAGS.log)
    tf.gfile.MakeDirs(FLAGS.log)
  except:
    print("Error creating log directory. Check permissions! Exiting...")
    quit()

  # predict
  ignore_crap = TRAIN["ignore_crap"]
  net.predict_dataset(FLAGS.dataset, path=FLAGS.path +
                      '/' + FLAGS.model, batchsize=FLAGS.batchsize,
                      ignore_last = ignore_crap)

  # clean up
  net.cleanup(None, None)
