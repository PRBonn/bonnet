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
  Extracts a log file from a graph, so as to be able to be read by
  tensorboard.
'''

import tensorflow as tf
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cnn_graph_log.py")
  parser.add_argument(
      '--log', '-l',
      type=str,
      required=True,
      help='Directory to log output of predictions.',
  )
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=True,
      help='Path to the graph. No default!'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Model Path: ", FLAGS.path)
  print("Log dir: ", FLAGS.log)
  print("----------\n")

  # define a graph
  g = tf.Graph()

  # fill it with the metagraph
  with g.as_default() as g:
    tf.train.import_meta_graph(FLAGS.path)

  # save the log from that graph
  with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir=FLAGS.log, graph=g)
