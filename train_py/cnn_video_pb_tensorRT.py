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
import skvideo.io as skio
import numpy as np

import dataset.aux_scripts.util as util

# tensorRT stuff
import pycuda.driver as cuda
import tensorrt as trt
from tensorrt.parsers import uffparser
import uff

# tensorflow stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up TF!
import tensorflow as tf


def predict_mask(cvim, frame, stream, d_input, d_output, context, FLAGS, DATA):
  # do all required transpositions
  cvim = cv2.cvtColor(cvim, cv2.COLOR_RGB2BGR)
  cvim = cv2.resize(cvim.astype(np.float32), (DATA['img_prop']['width'],
                                              DATA['img_prop']['height']),
                    interpolation=cv2.INTER_LINEAR)
  tcvim = np.transpose(cvim, axes=(2, 0, 1))
  tcvim = tcvim.copy(order='C')
  tcvim = (tcvim - 128.0) / 128.0

  # Bindings provided as pointers to the GPU memory.
  # PyCUDA lets us do this for memory allocations by
  # casting those allocations to ints
  bindings = [int(d_input), int(d_output)]

  # allocate memory on the CPU to hold results after inference
  output = np.empty((len(DATA['label_map']), DATA['img_prop']['height'],
                     DATA['img_prop']['width']), dtype=np.float32, order='C')

  # predict mask from image
  start = time.time()
  cuda.memcpy_htod_async(d_input, tcvim, stream)
  # execute model
  context.enqueue(1, bindings, stream.handle, None)
  # transfer predictions back
  cuda.memcpy_dtoh_async(output, d_output, stream)
  # syncronize threads
  stream.synchronize()
  elapsed = time.time() - start
  print("Prediction for frame ", frame, ". Elapsed: ", elapsed, "s")

  # mask from logits
  mask = np.argmax(output, axis=0)

  # change to color
  color_mask = util.prediction_to_color(mask, DATA["label_remap"],
                                        DATA["color_map"])

  # transparent
  im, transparent_mask = util.transparency(cvim, color_mask)
  all_img = np.concatenate((cvim, transparent_mask), axis=1)
  w, h, _ = all_img.shape
  watermark = "Time: {:.3f}s, FPS: {:.3f}img/s.".format(elapsed, 1 / elapsed)
  cv2.putText(all_img, watermark,
              org=(10, w - 10),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=0.75,
              color=(255, 255, 255),
              thickness=2,
              lineType=cv2.LINE_AA)

  # write to disk
  cv2.imwrite(FLAGS.log + "/mask_" + frame + ".jpg", color_mask)
  cv2.imwrite(FLAGS.log + "/full_" + frame + ".jpg", all_img)

  # show me the image
  cv2.imshow("video", all_img.astype(np.uint8))
  ch = cv2.waitKey(1)

  return ch


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./cnn_video_pb_tensorRT.py")
  parser.add_argument(
      '--video', '-v',
      type=str,
      required=False,
      default="",
      help='Video to infer.',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default='/tmp/pb_tRT_predictions/',
      help='Directory to log output of predictions. Defaults to %(default)s',
  )
  model_choices = ['FP32', 'FP16']
  parser.add_argument(
      '--precision',
      type=str,
      default='FP32',
      help='Precision for calculations (FP32, FP16). Default to %(default)s',
      choices=model_choices
  )
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=True,
      help='Directory to get the model. No default!'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Video to infer: ", FLAGS.video)
  print("Log dir: ", FLAGS.log)
  print("Precision: ", FLAGS.precision)
  print("model path", FLAGS.path)
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

  frozen_name = os.path.join(FLAGS.path, "optimized_tRT.pb")
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
  input_node = NODES["input_norm_and_resized_node"]
  mask_node = NODES["logits_node"]
  output_nodes = [mask_node]
  input_nodes = [input_node]

  # import uff from tensorflow frozen
  uff_model = uff.from_tensorflow_frozen_model(frozen_name,
                                               output_nodes,
                                               input_nodes=input_nodes)

  # creating a logger for TensorRT
  G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

  # create a uff parser
  parser = uffparser.create_uff_parser()
  parser.register_input(input_node, (DATA['img_prop']['depth'],
                                     DATA['img_prop']['height'],
                                     DATA['img_prop']['width']), 0)
  parser.register_output(mask_node)

  # pass the logger, parser and the uff model stream and some settings to create the engine
  MAX_ALLOWED_BATCH_SIZE = 1
  MAX_ALLOWED_WS_SIZE = 1 << 20
  if FLAGS.precision == "FP32":
    DATA_TYPE = trt.infer.DataType.FLOAT
  elif FLAGS.precision == "FP16":
    DATA_TYPE = trt.infer.DataType.HALF

  engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser,
                                       MAX_ALLOWED_BATCH_SIZE,
                                       MAX_ALLOWED_WS_SIZE,
                                       DATA_TYPE)  # .HALF for fp16 in jetson!

  # get rid of the parser
  parser.destroy()

  # create a runtime and an execution context for the engine
  runtime = trt.infer.create_infer_runtime(G_LOGGER)
  context = engine.create_execution_context()

  # alocate device memory
  input_size = DATA['img_prop']['depth'] * DATA['img_prop']['height'] * \
      DATA['img_prop']['width']
  output_size = len(DATA['label_map']) * DATA['img_prop']['height'] * \
      DATA['img_prop']['width']
  d_input = cuda.mem_alloc(1 * input_size * 4)
  d_output = cuda.mem_alloc(1 * output_size * 4)

  # cuda stream to run inference in.
  stream = cuda.Stream()

  # create resizeable window
  cv2.namedWindow("video", cv2.WINDOW_NORMAL)

  # open video capture
  if FLAGS.video is "":
    print("Webcam reading not implemented. Exiting")
    quit()
  else:
    inputparameters = {}
    outputparameters = {}
    reader = skio.FFmpegReader(FLAGS.video,
                               inputdict=inputparameters,
                               outputdict=outputparameters)

    i = 0
    for frame in reader.nextFrame():
      # predict
      ch = predict_mask(frame, str(i), stream, d_input,
                        d_output, context, FLAGS, DATA)
      if ch == 27:
        break
      # add to frame nr.
      i += 1
    # clean up
    cv2.destroyAllWindows()

  # Save the engine to a file to use later. Use this engine by using tensorrt.utils.load_engine
  trt.utils.write_engine_to_file(
      FLAGS.log + "pb-to-tRT.engine", engine.serialize())

  # Example use engine
  # new_engine = trt.utils.load_engine(G_LOGGER, "./tf_mnist.engine")

  # Clean up context, engine and runtime
  context.destroy()
  engine.destroy()
  runtime.destroy()
