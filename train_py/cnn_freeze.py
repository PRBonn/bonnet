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
  Takes a trained model in the training format and turns it into a frozen pb.
'''
import os
import argparse
import imp
import yaml
import signal
from shutil import copyfile

# tensorflow stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up TF!
import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cnn_freeze.py")
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
      '--log', '-l',
      type=str,
      default='/tmp/frozen_model',
      help='Directory to save the frozen graph. Defaults to %(default)s',
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
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("model path", FLAGS.path)
  print("model type", FLAGS.model)
  print("output dir", FLAGS.log)
  print("data yaml: ", FLAGS.data)
  print("net yaml: ", FLAGS.net)
  print("train yaml: ", FLAGS.train)
  print("Verbose?: ", FLAGS.verbose)
  print("----------\n")

  # try to open data yaml
  try:
    if(FLAGS.data):
      print("Opening desired data file %s" % FLAGS.data)
      f = open(FLAGS.data, 'r')
      datafile = FLAGS.data
    else:
      print("Opening default data file data.yaml from log folder")
      f = open(FLAGS.path + '/data.yaml', 'r')
      datafile = FLAGS.path + '/data.yaml'
    DATA = yaml.load(f)
  except:
    print("Error opening data yaml file...")
    quit()

  # try to open net yaml
  try:
    if(FLAGS.net):
      print("Opening desired net file %s" % FLAGS.net)
      f = open(FLAGS.net, 'r')
      netfile = FLAGS.net
    else:
      print("Opening default net file net.yaml from log folder")
      f = open(FLAGS.path + '/net.yaml', 'r')
      netfile = FLAGS.path + '/net.yaml'
    NET = yaml.load(f)
  except:
    print("Error opening net yaml file...")
    quit()

  # try to open train yaml
  try:
    if(FLAGS.train):
      print("Opening desired train file %s" % FLAGS.train)
      f = open(FLAGS.train, 'r')
      trainfile = FLAGS.train
    else:
      print("Opening default train file train.yaml from log folder")
      f = open(FLAGS.path + '/train.yaml', 'r')
      trainfile = FLAGS.path + '/train.yaml'
    TRAIN = yaml.load(f)
  except:
    print("Error opening train yaml file...")
    quit()

  if tf.gfile.Exists(FLAGS.path + '/' + FLAGS.model):
    print("model folder exists! Using model from %s" %
          (FLAGS.path + '/' + FLAGS.model))
  else:
    print("model folder does not exist. Gimme dat model yo!")
    quit()

  try:
    print("Creating log dir in", FLAGS.log)
    if tf.gfile.Exists(FLAGS.log):
      tf.gfile.DeleteRecursively(FLAGS.log)
    tf.gfile.MakeDirs(FLAGS.log)
  except:
    print("Error creating log directory. Check permissions! Exiting...")
    quit()

  # copy all files to log folder Also, standardize name to be able to open it later
  try:
    print("Copying files to %s for further reference." % FLAGS.log)
    copyfile(datafile, FLAGS.log + "/data.yaml")
    copyfile(netfile, FLAGS.log + "/net.yaml")
    copyfile(trainfile, FLAGS.log + "/train.yaml")
  except:
    print("Error copying files, check permissions. Exiting...")
    quit()

  # get architecture
  architecture = imp.load_source(
      "architecture", os.getcwd() + '/arch/' + NET["name"] + '.py')

  # build the network
  net = architecture.Network(DATA, NET, TRAIN, FLAGS.log)

  # handle ctrl-c for threads
  signal.signal(signal.SIGINT, net.cleanup)
  signal.signal(signal.SIGTERM, net.cleanup)

  # freeze the graph
  path = os.path.join(FLAGS.path, FLAGS.model)
  net.freeze_graph(path=path, verbose=FLAGS.verbose)

  # clean up
  net.cleanup(None, None)
