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
  App to train the desired architecture with the desired parameters, and on the
  desired dataset.
'''
# os and file stuff
import os
import argparse
import datetime
import imp
import yaml
from shutil import copyfile
import subprocess

# tensorflow stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up TF!
import tensorflow as tf
import signal

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cnn_train.py")
  parser.add_argument(
      '--data', '-d',
      type=str,
      required=False,
      help='Dataset yaml cfg file. See /cfg for sample. No default!',
  )
  parser.add_argument(
      '--net', '-n',
      type=str,
      required=False,
      help='Network yaml cfg file. See /cfg for sample. No default!',
  )
  parser.add_argument(
      '--train', '-t',
      type=str,
      required=False,
      help='Training hyperparameters yaml cfg file. See /cfg for sample. No default!',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the log data. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the model. If not passed, do not retrain!'
  )
  model_choices = ['acc', 'iou']
  parser.add_argument(
      '--model', '-m',
      type=str,
      default='iou',
      help='Type of model (best acc or best iou). Default to %(default)s',
      choices=model_choices
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("data yaml: ", FLAGS.data)
  print("net yaml: ", FLAGS.net)
  print("train yaml: ", FLAGS.train)
  print("log dir", FLAGS.log)
  print("model path", FLAGS.path)
  print("model type", FLAGS.model)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
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
    print("Error opening data yaml file. Check! Exiting...")
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
    print("Error opening net yaml file. Check! Exiting...")
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
    print("Error opening train yaml file. Check! Exiting...")
    quit()

  # create log folder
  try:
    if tf.gfile.Exists(FLAGS.log):
      tf.gfile.DeleteRecursively(FLAGS.log)
    tf.gfile.MakeDirs(FLAGS.log)
  except:
    print("Error creating log directory. Check permissions! Exiting...")
    quit()

  # does model folder exist?
  if FLAGS.path is not None:
    if tf.gfile.Exists(FLAGS.path + '/' + FLAGS.model):
      print("model folder exists! Using model from %s" %
            (FLAGS.path + '/' + FLAGS.model))
    else:
      print("model folder doesnt exist! Exiting...")
      quit()

  # copy all files to log folder (to remember what we did, and make inference
  # easier). Also, standardize name to be able to open it later
  try:
    print("Copying files to %s for further reference." % FLAGS.log)
    copyfile(datafile, FLAGS.log + "/data.yaml")
    copyfile(netfile, FLAGS.log + "/net.yaml")
    copyfile(trainfile, FLAGS.log + "/train.yaml")
  except:
    print("Error copying files, check permissions. Exiting...")
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

  # train
  if FLAGS.path is None:
    print("Training from scratch")
    net.train()
  else:
    print("Training from model in ", str(FLAGS.path + '/' + FLAGS.model))
    net.train(path=str(FLAGS.path + '/' + FLAGS.model))

  # clean up
  net.cleanup(None, None)
