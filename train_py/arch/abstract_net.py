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
  Network class, containing:
    - Training steps and training procedure
    - Checkpoint saver and restorer
    - Function to predict mask from image
    - etc :)

  API Style should be the same for all nets
'''

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.tools import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph
import numpy as np
import cv2
import imp
import os
import time
import sys
import yaml
import dataset.augment_data as ad
import dataset.aux_scripts.util as util
import arch.msg as msg


class AbstractNetwork:
  def __init__(self, DATA, NET, TRAIN, logdir):
    # init
    self.DATA = DATA      # dictionary with dataset parameters
    self.NET = NET        # dictionary with network parameters
    self.TRAIN = TRAIN    # dictionary with training hyperparams
    self.log = logdir     # where to put the log for training
    self.sess = None      # session (no session until needed)
    self.code_valid = None  # if this is not defined in the graph, we need to complain

  def build_graph(self, train_stage, data_format="NCHW"):
    # some graph info depending on what I will do with it
    print("This needs to be re-implemented in each arch. Exiting...")
    quit()
    return

  def resize_label(self, lbls_pl):
    """ Resize the y pl to fit the image for loss and confusion matrix
    """
    # reshape label
    lbls_pl_exp = tf.expand_dims(lbls_pl, -1)
    lbls_resized = tf.image.resize_images(lbls_pl_exp,
                                          [self.DATA["img_prop"]["height"],
                                           self.DATA["img_prop"]["width"]],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lbls_resized = tf.reshape(lbls_resized, [self.batch_size_gpu,
                                             self.DATA["img_prop"]["height"],
                                             self.DATA["img_prop"]["width"]])
    return lbls_resized

  def loss_f(self, lbls_pl, logits_train, gamma_focal=2, w_t="log", w_d=1e-4):
    """Calculates the loss from the logits and the labels.
    """
    print("Defining loss function")
    with tf.variable_scope("loss"):
      lbls_resized = self.resize_label(lbls_pl)

      # Apply median freq balancing (median frec / freq(class))
      w = np.empty(len(self.dataset.train.content))

      if w_t == "log":
        # get the frequencies and weights
        for key in self.dataset.train.content:
          e = 1.02  # max weight = 50
          f_c = self.dataset.train.content[key]
          w[self.DATA["label_remap"][key]] = 1 / np.log(f_c + e)
        print("\nWeights for loss function (1/log(frec(c)+e)):\n", w)

      elif w_t == "median_freq":
        # get the frequencies
        f = np.empty(len(self.dataset.train.content))
        for key in self.dataset.train.content:
          e = 0.001
          f_c = self.dataset.train.content[key]
          f[self.DATA["label_remap"][key]] = f_c
          w[self.DATA["label_remap"][key]] = 1 / (f_c + e)

        # calculate the median frequencies and normalize
        median_freq = np.median(f)
        print("\nFrequencies of classes:\n", f)
        print("\nMedian freq:\n", median_freq)
        print("\nWeights for loss function (1/frec(c)):\n", w)
        w = median_freq * w
        print("\nWeights for loss function (median frec/frec(c)):\n", w)
      else:
        print("Using natural weights, since no valid loss option was given.")
        w.fill(1.0)
        for key in self.dataset.train.content:
          if self.dataset.train.content[key] == float("inf"):
            w[self.DATA["label_remap"][key]] = 0
        print("weights: ", w)

      # use class weights as tf constant
      w_tf = tf.constant(w, dtype=tf.float32, name='class_weights')
      w_mask = w.astype(np.bool).astype(np.float32)
      w_mask_tf = tf.constant(w_mask, dtype=tf.float32,
                              name='class_weights_mask')

      # make logits softmax matrixes for loss
      loss_epsilon = tf.constant(value=1e-10)
      softmax = tf.nn.softmax(logits_train)
      softmax_mat = tf.reshape(softmax, (-1, self.num_classes))
      zerohot_softmax_mat = 1 - softmax_mat

      # make the labels one-hot for the cross-entropy
      onehot_mat = tf.reshape(tf.one_hot(lbls_resized, self.num_classes),
                              (-1, self.num_classes))

      # make the zero hot to punish the false negatives, but ignore the
      # zero-weight classes
      masked_sum = tf.reduce_sum(onehot_mat * w_mask_tf, axis=1)
      zeros = onehot_mat * 0.0
      zerohot_mat = tf.where(tf.less(masked_sum, 1e-5),
                             x=zeros,
                             y=1 - onehot_mat)

      # focal loss p and gamma
      gamma = np.full(onehot_mat.get_shape().as_list(), fill_value=gamma_focal)
      gamma_tf = tf.constant(gamma, dtype=tf.float32)
      focal_softmax = tf.pow(1 - softmax_mat, gamma_tf) * \
          tf.log(softmax_mat + loss_epsilon)
      zerohot_focal_softmax = tf.pow(1 - zerohot_softmax_mat, gamma_tf) * \
          tf.log(zerohot_softmax_mat + loss_epsilon)

      # calculate xentropy
      cross_entropy = - tf.reduce_sum(tf.multiply(focal_softmax * onehot_mat +
                                                  zerohot_focal_softmax * zerohot_mat, w_tf),
                                      axis=[1])

      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

      # weight decay
      print("Weight decay: ", w_d)
      w_d_tf = tf.constant(w_d, dtype=tf.float32, name='weight_decay')
      variables = tf.trainable_variables(scope="model")
      for var in variables:
        if "weights" in var.name:
          loss += w_d_tf * tf.nn.l2_loss(var)
      return loss

  def average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    This function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers. Notice that this function already averages the gradients,
       it doesn't sum them. This is important when scaling the hyper-params for
       multi-gpu training.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # the variables are redundant because they are shared across towers.
      # So we just return the first tower's pointer to the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)

    return average_grads

  def restore_session(self, path):
    # restore from checkpoint (to continue training, or to infer at test time)
    print("Restoring checkpoint")

    # Restore the graph
    print("Looking for model in %s" % path)
    self.ckpt = tf.train.get_checkpoint_state(path)

    # only try if I have a checkpoint
    if self.ckpt and self.ckpt.model_checkpoint_path:
      print("Retrieving model from: ", self.ckpt.model_checkpoint_path)

      # try to get the full model including classifier, but with no crap from
      # previous training such as learning rate, moments, etc.
      try:
        restore = []
        not_restore = []
        restore.extend(tf.global_variables(scope='model'))
        restore_var = [v for v in restore if v not in not_restore]
        restore_saver = tf.train.Saver(var_list=restore_var)
        # restore all variables
        restore_saver.restore(self.sess, self.ckpt.model_checkpoint_path)

      except:
        # if it fails to load, reload only the feat extractor, and not the linear
        # classifier. This is useful when retraining for a different number of classes
        print(' WARNING '.center(80, '*'))
        print("Failed to restore model".center(80, '!'))
        print('*' * 80)
        print("Keeping classifier random, to see if this helps (also keeping all the training stuff the same)")
        restore = []
        not_restore = []
        restore.extend(tf.global_variables(scope='model'))
        not_restore.extend(tf.global_variables(scope='model/logits'))
        restore_var = [v for v in restore if v not in not_restore]
        restore_saver = tf.train.Saver(var_list=restore_var)
        # restore all variables
        restore_saver.restore(self.sess, self.ckpt.model_checkpoint_path)

        try:
          # try again without the linear part
          restore_saver.restore(self.sess, self.ckpt.model_checkpoint_path)

        except:
          # if all fails, I need to be doing something wrong, like using
          # a wrong arch checkpoint. Report and exit
          print("Restore failed again. Something else is wrong. Exiting")
          quit()

      # hooray! Everything great
      print("Successfully restored model weights! :D")
      return True

    else:
      # no model :(
      print("No model to restore in path")
      return False

  def predict_kickstart(self, path, batchsize=1, data_format="NCHW"):
    # bake placeholders
    self.img_pl, self.lbls_pl = self.placeholders(
        self.DATA["img_prop"]["depth"], batchsize)

    # make list
    self.n_gpus = 1
    self.img_pl_list = [self.img_pl]
    self.lbls_pl_list = [self.lbls_pl]

    # inititialize inference graph
    print("Initializing network")
    with tf.name_scope("test_model"):
      with tf.variable_scope("model", reuse=None):
        self.logits_valid, self.code_valid, self.n_img_valid = self.build_graph(
            self.img_pl, False, data_format=data_format)  # not training

    # lists of outputs
    self.logits_valid_list = [self.logits_valid]
    self.logits_code_list = [self.code_valid]

    # get model size and report it (so that I can report in paper)
    n_parameters = 0
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
      # print(var.name , var.get_shape().as_list(), np.prod(var.get_shape().as_list()))
      var_params = np.prod(var.get_shape().as_list())
      n_parameters += var_params
    print("*" * 80)
    print("Total number of parameters in network: ",
          "{:,}".format(n_parameters))
    print("*" * 80)

    # build graph and predict value (if graph is not built)
    print("Predicting mask")

    # set up evaluation head in the graph
    with tf.variable_scope("output"):
      self.output_p = tf.nn.softmax(self.logits_valid)
      self.mask = tf.argmax(self.output_p, axis=3, output_type=tf.int32)

    # report the mask shape as a sort of sanity check
    mask_shape = self.mask.get_shape().as_list()
    print("mask shape", mask_shape)

    # metadata collector for verbose mode (spits out layer-wise profile)
    self.run_metadata = tf.RunMetadata()

    # Add the variable initializer Op.
    self.init = tf.global_variables_initializer()

    # Create a saver for restoring and saving checkpoints.
    self.saver = tf.train.Saver(save_relative_paths=True)

    # xla stuff for faster inference (and soft placement for low ram device)
    gpu_options = tf.GPUOptions(allow_growth=True, force_gpu_compatible=True)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False, gpu_options=gpu_options)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2

    # start a session
    self.sess = tf.Session(config=config)

    # init variables
    self.sess.run(self.init)

    # if path to model is give, try to restore:
    self.restore_session(path)

    print("Saving this graph in %s" % self.log)
    self.summary_writer = tf.summary.FileWriter(self.log, self.sess.graph)
    self.summary_writer.flush()

    # save this graph
    self.chkpt_graph = os.path.join(self.log, 'model.ckpt')
    self.saver.save(self.sess, self.chkpt_graph)
    tf.train.write_graph(self.sess.graph_def, self.log, 'model.pbtxt')

  def freeze_graph(self, path=None, verbose=False):
    """ Extract the sub graph defined by the output nodes and convert
        all its variables into constant
    """
    # kickstart the model. If session is initialized everything may be dirty,
    # so please use this function from a clean tf environment :)
    if self.sess is None:
      self.predict_kickstart(path, data_format="NHWC")
    else:
      print("existing session. This is unintended behavior. Check!")
      quit()

    # outputs
    in_node_names = [str(self.img_pl.op.name)]
    print("in_node_names", in_node_names)
    in_trt_node_names = [str(self.n_img_valid.op.name)]
    print("in_tensorRT_node_names", in_trt_node_names)
    out_node_names = [str(self.mask.op.name), str(self.code_valid.op.name)]
    print("out_node_names", out_node_names)
    input_graph_path = os.path.join(self.log, 'model.pbtxt')
    checkpoint_path = os.path.join(self.log, 'model.ckpt')
    input_saver_def_path = ""
    input_binary = False
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    out_frozen_graph_name_nchw = os.path.join(self.log, 'frozen_nchw.pb')
    out_frozen_graph_name_nhwc = os.path.join(self.log, 'frozen_nhwc.pb')
    out_opt_graph_name = os.path.join(self.log, 'optimized.pb')
    out_opt_tensorRT_graph_name = os.path.join(self.log, 'optimized_tRT.pb')
    uff_opt_tensorRT_graph_name = os.path.join(self.log, 'optimized_tRT.uff')
    output_quantized_graph_name = os.path.join(self.log, 'quantized.pb')
    clear_devices = True

    # freeze
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, ",".join(
                                  out_node_names),
                              restore_op_name, filename_tensor_name,
                              out_frozen_graph_name_nhwc, clear_devices, "")

    # Optimize for inference
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(out_frozen_graph_name_nhwc, "rb") as f:
      data = f.read()
      input_graph_def.ParseFromString(data)

    # transforms for optimization
    transforms = ['add_default_attributes',
                  'remove_nodes(op=Identity, op=CheckNumerics)',
                  'fold_constants(ignore_errors=true)', 'fold_batch_norms',
                  'fold_old_batch_norms',
                  'strip_unused_nodes', 'sort_by_execution_order']

    # optimize and save
    output_graph_def = TransformGraph(input_graph_def,
                                      in_node_names,
                                      out_node_names,
                                      transforms)
    f = tf.gfile.FastGFile(out_opt_graph_name, "w")
    f.write(output_graph_def.SerializeToString())

    # quantize and optimize, and save
    transforms += ['quantize_weights', 'quantize_nodes']
    output_graph_def = TransformGraph(input_graph_def,
                                      in_node_names,
                                      out_node_names,
                                      transforms)
    f = tf.gfile.FastGFile(output_quantized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())

    # save the names of the input and output nodes
    input_node = str(self.img_pl.op.name)
    input_norm_and_resized_node = str(self.n_img_valid.op.name)
    code_node = str(self.code_valid.op.name)
    logits_node = str(self.logits_valid.op.name)
    out_probs_node = str(self.output_p.op.name)
    mask_node = str(self.mask.op.name)
    node_dict = {"input_node": input_node,
                 "input_norm_and_resized_node": input_norm_and_resized_node,
                 "code_node": code_node,
                 "logits_node": logits_node,
                 "out_probs_node": out_probs_node,
                 "mask_node": mask_node}
    node_file = os.path.join(self.log, "nodes.yaml")
    with open(node_file, 'w') as f:
      yaml.dump(node_dict, f, default_flow_style=False)

    # do the same for NCHW but don't save any quantized models,
    # since quantization doesn't work in NCHW (only save optimized for tensort)
    self.sess.close()
    tf.reset_default_graph()
    self.predict_kickstart(path, data_format="NCHW")

    # freeze
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path,
                              ",".join(out_node_names),
                              restore_op_name, filename_tensor_name,
                              out_frozen_graph_name_nchw, clear_devices, "")

    # Optimize for inference on tensorRT
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(out_frozen_graph_name_nchw, "rb") as f:
      data = f.read()
      input_graph_def.ParseFromString(data)

    # transforms for optimization
    transforms = ['add_default_attributes',
                  'remove_nodes(op=Identity, op=CheckNumerics)',
                  'fold_batch_norms', 'fold_old_batch_norms',
                  'strip_unused_nodes', 'sort_by_execution_order']

    # optimize and save
    output_graph_def = TransformGraph(input_graph_def,
                                      in_trt_node_names,
                                      out_node_names,
                                      transforms)
    f = tf.gfile.FastGFile(out_opt_tensorRT_graph_name, "w")
    f.write(output_graph_def.SerializeToString())
    f.close()

    # last but not least, try to convert the NCHW model to UFF for TensorRT
    # inference
    print("Saving uff model for TensorRT inference")
    try:
      # import tensorRT stuff
      import uff
      # import uff from tensorflow frozen and save as uff file
      uff.from_tensorflow_frozen_model(out_opt_tensorRT_graph_name,
                                       [logits_node],
                                       input_nodes=[
                                           input_norm_and_resized_node],
                                       output_filename=uff_opt_tensorRT_graph_name)
    except:
      print("Error saving TensorRT UFF model")

    return

  def gpu_available(self):
    # can I use a gpu? Return number of GPUs available.
    # tensorflow is very greedy with the GPUs, and it always tries to use
    # everything available. So make sure you restrict its vision with
    # the CUDA_VISIBLE_DEVICES environment variable.
    n_gpus_avail = 0
    devices = device_lib.list_local_devices()
    for dev in devices:
      print("DEVICE AVAIL: ", dev.name)
      if '/device:GPU' in dev.name:
        n_gpus_avail += 1
    return n_gpus_avail

  def predict(self, img, path=None, verbose=False, as_probs=False):
    ''' Predict an opencv image labels with a trained model. Kickstarts the
        session if it is the first call
    '''

    # if there is no session, kick it!
    if self.sess is None:
      # get dataset reader
      print("Fetching dataset")
      self.parser = imp.load_source("parser",
                                    os.getcwd() + '/dataset/' +
                                    self.DATA["name"] + '.py')

      # kickstart in NCHW or NHWC depending on availability or not of GPUs
      n_gpus_avail = self.gpu_available()
      if n_gpus_avail:
        self.predict_kickstart(path, data_format="NCHW")
      else:
        self.predict_kickstart(path, data_format="NHWC")

    # choose op to run according to choice of mask or feature map:
    if as_probs:
      node_to_run = self.output_p
    else:
      node_to_run = self.mask

    # run the classifier and report according to desired verbosity
    if verbose:
      # run the classifier in verbose mode (get profile and report it)
      start_time = time.time()
      predicted_mask = self.sess.run(node_to_run, {self.img_pl: [img]},
                                     options=tf.RunOptions(
          trace_level=tf.RunOptions.FULL_TRACE),
          run_metadata=self.run_metadata)
      time_to_run = time.time() - start_time
      print("Time to evaluate: %f" % time_to_run)

      # profile amount of flops
      opts = tf.profiler.ProfileOptionBuilder.float_operation()
      flops = tf.profiler.profile(tf.get_default_graph(
      ), run_meta=self.run_metadata, cmd='op', options=opts)
      if flops is not None:
        print("*" * 80)
        print("Amount of floating point ops (FLOPs): ",
              "{:,}".format(flops.total_float_ops))
        print("*" * 80)

      # Builder to create options to profile the time and memory information.
      builder = tf.profiler.ProfileOptionBuilder

      # profile with stdout
      opts = (builder(builder.time_and_memory()).with_stdout_output().build())
      tf.profiler.profile(tf.get_default_graph(),
                          run_meta=self.run_metadata, cmd='op', options=opts)

      # profile with log file
      tracename = os.path.join(self.log, 'timeline.ctf.json')
      opts = (builder(builder.time_and_memory()
                      ).with_timeline_output(tracename).build())
      tf.profiler.profile(tf.get_default_graph(),
                          run_meta=self.run_metadata, cmd='graph', options=opts)

    else:
      # run the classifier and report nothing back!
      predicted_mask = self.sess.run(node_to_run, {self.img_pl: [img]})

    # return the single prediction
    return predicted_mask[0]

  def predict_code(self, img, path=None, verbose=False):
    ''' Extract CNN features from an opencv image with a trained model.
        Kickstarts the session if it is the first call.
    '''

    if self.sess is None:
      # get dataset reader
      print("Fetching dataset")
      self.parser = imp.load_source("parser",
                                    os.getcwd() + '/dataset/' +
                                    self.DATA["name"] + '.py')
      # kickstart in NCHW or NHWC depending on availability or not of GPUs
      n_gpus_avail = self.gpu_available()
      if n_gpus_avail:
        self.predict_kickstart(path, data_format="NCHW")
      else:
        self.predict_kickstart(path, data_format="NHWC")

    # check if arch gave me the code in the kickstarting
    if self.code_valid is None:
      print("Code is not defined in architecture. Can't be inferred.")
      quit()

    # run the feature extractor and report back according to verbosity
    if verbose:
      start_time = time.time()
      infered_code = self.sess.run(self.code_valid, {self.img_pl: [img]},
                                   options=tf.RunOptions(
          trace_level=tf.RunOptions.FULL_TRACE),
          run_metadata=self.run_metadata)
      time_to_run = time.time() - start_time
      print("Time to evaluate: %f" % time_to_run)

      # profile amount of flops
      opts = tf.profiler.ProfileOptionBuilder.float_operation()
      flops = tf.profiler.profile(tf.get_default_graph(
      ), run_meta=self.run_metadata, cmd='op', options=opts)
      if flops is not None:
        print("*" * 80)
        print("Amount of floating point ops (FLOPs): ",
              "{:,}".format(flops.total_float_ops))
        print("*" * 80)

      # Builder to create options to profile the time and memory information.
      builder = tf.profiler.ProfileOptionBuilder

      # profile with stdout
      opts = (builder(builder.time_and_memory()).with_stdout_output().build())
      tf.profiler.profile(tf.get_default_graph(),
                          run_meta=self.run_metadata, cmd='op', options=opts)

      # profile with log file
      tracename = os.path.join(self.log, 'timeline.ctf.json')
      opts = (builder(builder.time_and_memory()
                      ).with_timeline_output(tracename).build())
      tf.profiler.profile(tf.get_default_graph(),
                          run_meta=self.run_metadata, cmd='graph', options=opts)

    else:
      infered_code = self.sess.run(self.code_valid, {self.img_pl: [img]})

    # return the single feature map as 3D numpy array
    return infered_code[0]

  def predict_dataset(self, datadir, path, batchsize=1, ignore_last=False):
    ''' Test accuracy in an entire dataset. Also kickstarts the session if needed
    '''
    if self.sess is None:
      # get dataset reader
      print("Fetching dataset")
      self.parser = imp.load_source("parser",
                                    os.getcwd() + '/dataset/' +
                                    self.DATA["name"] + '.py')
      # import dataset
      self.DATA["data_dir"] = datadir
      self.dataset = self.parser.read_data_sets(self.DATA)

      # define mode of model according to gpu availability
      n_gpus_avail = self.gpu_available()
      if n_gpus_avail:
        self.predict_kickstart(path, data_format="NCHW")
      else:
        self.predict_kickstart(path, data_format="NHWC")

    # run the classifier in each split of dataset
    print("Train data")
    self.dataset_accuracy(self.dataset.train, batchsize, ignore_last)
    print("//////////\n\n")
    print("Validation data")
    self.dataset_accuracy(self.dataset.validation, batchsize, ignore_last)
    print("//////////\n\n")
    print("Test data")
    self.dataset_accuracy(self.dataset.test, batchsize, ignore_last)

    return

  def pix_histogram(self, mask, lbl):
    '''
      get individual mask and label and create 2d hist
    '''
    # flatten mask and cast
    flat_mask = mask.flatten().astype(np.uint32)
    # flatten label and cast
    flat_label = lbl.flatten().astype(np.uint32)
    # get the histogram
    histrange = np.array([[-0.5, self.num_classes - 0.5],
                          [-0.5, self.num_classes - 0.5]], dtype='float64')
    h_now, _, _ = np.histogram2d(np.array(flat_mask),
                                 np.array(flat_label),
                                 bins=self.num_classes,
                                 range=histrange)
    return h_now

  def pix_acc_from_histogram(self, hist):
    '''
      get complete 2d hist and return:
        mean accuracy
        per class iou
        mean iou
        per class precision
        per class recall
    '''
    # calculate accuracy from histogram
    if hist.sum():
      mean_acc = np.diag(hist).sum() / hist.sum()
    else:
      mean_acc = 0

    # calculate IoU
    per_class_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_iou = np.nanmean(per_class_iou)

    # calculate precision and recall
    per_class_prec = np.diag(hist) / hist.sum(1)
    per_class_rec = np.diag(hist) / hist.sum(0)

    return mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec

  def obj_histogram(self, mask, label):
    # holders for predicted object and right object (easily calculate histogram)
    predicted = []
    labeled = []

    # get connected components in label for each class
    for i in range(self.num_classes):
      # get binary image for this class
      bin_lbl = np.zeros(label.shape)
      bin_lbl[label == i] = 1
      bin_lbl[label != i] = 0

      # util.im_gray_plt(bin_lbl,'class '+str(i))
      connectivity = 4
      output = cv2.connectedComponentsWithStats(
          bin_lbl.astype(np.uint8), connectivity, cv2.CV_32S)
      num_components = output[0]
      components = output[1]
      stats = output[2]
      centroids = output[3]

      for j in range(1, num_components):  # 0 is background (useless)
        # only process if it has more than 50pix
        if stats[j][cv2.CC_STAT_AREA] > 50:
          # for each component in each class, see the class with the highest percentage of pixels
          # make mask with just this component of this class
          comp_mask = np.zeros(label.shape)
          comp_mask[components == j] = 0
          comp_mask[components != j] = 1
          # mask the prediction
          masked_prediction = np.ma.masked_array(mask, mask=comp_mask)
          # get histogram and get the argmax that is not zero
          class_hist, _ = np.histogram(masked_prediction.compressed(),
                                       bins=self.num_classes, range=[0, self.num_classes])
          max_class = np.argmax(class_hist)
          # print("\nMax class: ",max_class,"  real: ",i)
          # util.im_gray_plt(comp_mask)
          # util.im_block()
          # sum an entry to the containers depending on right or wrong
          predicted.append(max_class)
          labeled.append(i)
    # for idx in range(len(predicted)):
    #   print(predicted[idx],labeled[idx])

    # histogram to count right and wrong objects
    histrange = np.array([[-0.5, self.num_classes - 0.5],
                          [-0.5, self.num_classes - 0.5]], dtype='float64')
    h_now, _, _ = np.histogram2d(np.array(predicted),
                                 np.array(labeled),
                                 bins=self.num_classes,
                                 range=histrange)

    return h_now

  def obj_acc_from_histogram(self, hist):
    # calculate accuracy, precision and recall
    if hist.sum():
      obj_acc = np.diag(hist).sum() / hist.sum()
    else:
      obj_acc = 0

    # calculate precision and recall
    obj_prec = np.diag(hist) / hist.sum(1)
    obj_rec = np.diag(hist) / hist.sum(0)

    return obj_acc, obj_prec, obj_rec

  def individual_accuracy(self, mask, label):
    # individual image prediction accuracy with label

    # check size of label
    proper_w = self.DATA["img_prop"]["width"]
    proper_h = self.DATA["img_prop"]["height"]
    h, w = label.shape
    if proper_w != w or proper_h != h:
      label = ad.resize(label, [proper_h, proper_w], neighbor=True)

    # calculate pixelwise accuracy from histogram
    hist = self.pix_histogram(mask, label)
    mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec = self.pix_acc_from_histogram(
        hist)
    print(" Pixelwise Performance: ")
    print('   Mean Accuracy: %0.04f, Mean IoU: %0.04f' % (mean_acc, mean_iou))
    print('   Intersection over union:')
    for idx in range(0, len(per_class_iou)):
      print('     class %d IoU: %f' % (idx, per_class_iou[idx]))
    print('   Precision:')
    for idx in range(0, len(per_class_prec)):
      print('     class %d Precision: %f' % (idx, per_class_prec[idx]))
    print('   Recall:')
    for idx in range(0, len(per_class_rec)):
      print('     class %d Recall: %f' % (idx, per_class_rec[idx]))

    # report objectwise accuracy
    hist = self.obj_histogram(mask, label)
    obj_acc, obj_prec, obj_rec = self.obj_acc_from_histogram(hist)
    print(" Objectwise Performance: ")
    print('   Accuracy: %0.04f' % (obj_acc))
    print('   Precision:')
    for idx in range(0, len(obj_prec)):
      print('     class %d Precision: %f' % (idx, obj_prec[idx]))
    print('   Recall:')
    for idx in range(0, len(obj_rec)):
      print('     class %d Recall: %f' % (idx, obj_rec[idx]))

    return mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec, obj_acc, obj_prec, obj_rec

  def dataset_accuracy(self, dataset, batch_size, ignore_last=False):
    ''' Slower metrics using numpy confusion matrix, and reporting estimate
        objectwise metrics, for testing
    '''

    # define accuracy metric for this model
    start_time_overall = time.time()  # save curr time to report duration
    inference_time = 0.0
    steps_per_epoch = dataset.num_examples // batch_size
    assert(steps_per_epoch > 0 and "Dataset length should be more than batchsize")
    num_examples = steps_per_epoch * batch_size
    pix_hist = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
    obj_hist = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
    for step in range(steps_per_epoch):
      feed_dict, names = self.fill_feed_dict(
          dataset, self.img_pl_list, self.lbls_pl_list, batch_size)
      for g in range(0, self.n_gpus):
        inference_start = time.time()
        pred = self.sess.run(self.logits_valid_list[g], feed_dict=feed_dict)
        inference_time += time.time() - inference_start
        # calculate 2d histogram of size (n_classes,n_classes)
        # one axis is the true label, the other one the predicted value, so
        # the diagonal contains the right detections
        for idx in range(0, batch_size):
          # get mask and labels
          mask = pred[idx].argmax(2)
          img = feed_dict[self.img_pl_list[g]][idx]
          label = feed_dict[self.lbls_pl_list[g]][idx]
          name = names[g][idx]
          if ".png" in name:
            name = name.replace(".png", ".jpg")
          # check size of label
          proper_w = self.DATA["img_prop"]["width"]
          proper_h = self.DATA["img_prop"]["height"]
          h, w = label.shape
          if proper_w != w or proper_h != h:
            label = ad.resize(label, [proper_h, proper_w], neighbor=True)
            img = ad.resize(img, [proper_h, proper_w])
          # get histograms
          pix_h_now = self.pix_histogram(mask, label)
          obj_h_now = self.obj_histogram(mask, label)
          # sum to history
          pix_hist += pix_h_now
          obj_hist += obj_h_now
          if self.TRAIN["save_imgs"]:
            color_mask = util.prediction_to_color(
                mask, self.DATA["label_remap"], self.DATA["color_map"])
            color_label = util.prediction_to_color(
                label, self.DATA["label_remap"], self.DATA["color_map"])
            path_to_save = self.log + '/predictions/'
            if not tf.gfile.Exists(path_to_save):
              tf.gfile.MakeDirs(path_to_save)
            cv2.imwrite(path_to_save + dataset.name + '_' + str(name),
                        np.concatenate((img, color_mask, color_label), axis=1))

    # calculate pixelwise metrics  histogram
    if ignore_last:
      pix_hist = pix_hist[:-1, :-1]
    mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec = self.pix_acc_from_histogram(
        pix_hist)

    # calculate objectwise metrics from histogram
    if ignore_last:
      obj_hist = obj_hist[:-1, :-1]
    obj_acc, obj_prec, obj_rec = self.obj_acc_from_histogram(obj_hist)

    overall_duration = time.time() - start_time_overall  # calculate time elapsed
    print('   Num samples: %d, Time to run %.3f sec (only inference: %.3f sec)' %
          (num_examples, overall_duration, inference_time))
    fps = (num_examples / inference_time)
    print('   Network FPS: %.3f' % fps)
    print('   Time per image: %.3f s' % (1 / fps))
    print(" Pixelwise Performance: ")
    print('   Mean Accuracy: %0.04f, Mean IoU: %0.04f' % (mean_acc, mean_iou))
    print('   Intersection over union:')
    for idx in range(0, len(per_class_iou)):
      print('     class %d IoU: %f' % (idx, per_class_iou[idx]))
    print('   Precision:')
    for idx in range(0, len(per_class_prec)):
      print('     class %d Precision: %f' % (idx, per_class_prec[idx]))
    print('   Recall:')
    for idx in range(0, len(per_class_rec)):
      print('     class %d Recall: %f' % (idx, per_class_rec[idx]))

    print(" Objectwise Performance: ")
    print('   Accuracy: %0.04f' % (obj_acc))
    print('   Precision:')
    for idx in range(0, len(obj_prec)):
      print('     class %d Precision: %f' % (idx, obj_prec[idx]))
    print('   Recall:')
    for idx in range(0, len(obj_rec)):
      print('     class %d Recall: %f' % (idx, obj_rec[idx]))

    return mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec

  def training_dataset_accuracy(self, dataset, batch_size, batch_size_gpu,
                                ignore_last=False):
    ''' Faster tensorflow metrics using tensorflow confusion matrix,
        for training
    '''

    # define accuracy metric for this model
    start_time_overall = time.time()  # save curr time to report duration
    inference_time = 0.0
    steps_per_epoch = dataset.num_examples // batch_size
    assert(steps_per_epoch > 0 and "Dataset length should be more than batchsize")
    num_examples = steps_per_epoch * batch_size
    pix_hist = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)
    for step in range(steps_per_epoch):
      feed_dict, names = self.fill_feed_dict(
          dataset, self.img_pl_list, self.lbls_pl_list, batch_size_gpu)
      inference_start = time.time()
      pix_h_now, pred = self.sess.run(
          [self.confusion_matrix, self.logits_valid], feed_dict=feed_dict)
      inference_time += time.time() - inference_start
      # masks from logits
      masks = pred.argmax(3)
      # sum to history
      pix_hist += pix_h_now
      # save to disk
      for g in range(0, self.n_gpus):
        for idx in range(0, batch_size_gpu):
          # get mask and labels
          img = feed_dict[self.img_pl_list[g]][idx]
          label = feed_dict[self.lbls_pl_list[g]][idx]
          name = names[g][idx]
          mask = masks[idx + g * batch_size_gpu]
          if ".png" in name:
            name = name.replace(".png", ".jpg")
          # check size of label
          proper_w = self.DATA["img_prop"]["width"]
          proper_h = self.DATA["img_prop"]["height"]
          h, w = label.shape

          # save predictions
          if self.TRAIN["save_imgs"]:
            # resize if proper
            if proper_w != w or proper_h != h:
              label = ad.resize(label, [proper_h, proper_w], neighbor=True)
              img = ad.resize(img, [proper_h, proper_w])
            # convert to color
            color_mask = util.prediction_to_color(
                mask, self.DATA["label_remap"], self.DATA["color_map"])
            color_label = util.prediction_to_color(
                label, self.DATA["label_remap"], self.DATA["color_map"])
            path_to_save = self.log + '/predictions/'
            if not tf.gfile.Exists(path_to_save):
              tf.gfile.MakeDirs(path_to_save)
            cv2.imwrite(path_to_save + dataset.name + '_' + str(name),
                        np.concatenate((img, color_mask, color_label), axis=1))

    # calculate pixelwise metrics histogram
    if ignore_last:
      pix_hist = pix_hist[:-1, :-1]
    mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec = self.pix_acc_from_histogram(
        pix_hist)

    overall_duration = time.time() - start_time_overall  # calculate time elapsed
    print('   Num samples: %d, Time to run %.3f sec (only inference: %.3f sec)' %
          (num_examples, overall_duration, inference_time))
    fps = (num_examples / inference_time)
    print('   Network FPS: %.3f' % fps)
    print('   Time per image: %.3f s' % (1 / fps))
    print(" Pixelwise Performance: ")
    print('   Mean Accuracy: %0.04f, Mean IoU: %0.04f' % (mean_acc, mean_iou))
    print('   Intersection over union:')
    for idx in range(0, len(per_class_iou)):
      print('     class %d IoU: %f' % (idx, per_class_iou[idx]))
    print('   Precision:')
    for idx in range(0, len(per_class_prec)):
      print('     class %d Precision: %f' % (idx, per_class_prec[idx]))
    print('   Recall:')
    for idx in range(0, len(per_class_rec)):
      print('     class %d Recall: %f' % (idx, per_class_rec[idx]))

    return mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec

  def assign_to_device(self, op_dev, var_dev='/cpu:0'):
    """Returns a function to place variables on the var_dev, and the ops in the
    op_dev.

    Args:
      op_dev: Device for ops
      var_dev: Device for variables
    """
    VAR_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable',
               'MutableHashTable', 'MutableHashTableOfTensors',
               'MutableDenseHashTable']

    def _assign(op):
      node_def = op if isinstance(op, tf.NodeDef) else op.node_def
      if node_def.op in VAR_OPS:
        return "/" + var_dev
      else:
        return op_dev
    return _assign

  def train(self, path=None):
    ''' Main function to train a network from scratch or from checkpoint
    '''

    # get dataset reader
    print("Fetching dataset")
    self.parser = imp.load_source("parser",
                                  os.getcwd() + '/dataset/' +
                                  self.DATA["name"] + '.py')

    # report batch size and gpus to use
    self.batch_size = int(self.TRAIN["batch_size"])
    self.n_gpus = int(self.TRAIN['gpus'])
    print("Training with %d GPU's" % self.n_gpus)
    print("Training with batch size %d" % self.batch_size)

    # gpus available
    self.n_gpus_avail = self.gpu_available()
    print("Number of GPU's available is %d" % self.n_gpus_avail)
    assert(self.n_gpus == self.n_gpus_avail)

    # calculate batch size per gpu
    self.batch_size_gpu = int(self.batch_size / self.n_gpus_avail)
    assert(self.batch_size % self.n_gpus == 0)
    assert(self.batch_size_gpu > 0)
    print("This means %d images per GPU" % self.batch_size_gpu)

    # import dataset
    self.dataset = self.parser.read_data_sets(self.DATA)

    # get learn rate from config file
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      self.lrate = self.TRAIN['lr']
      with tf.variable_scope("learn_rate"):
        lr_init = tf.constant(self.lrate)
        self.learn_rate_var = tf.get_variable(name="learn_rate",
                                              initializer=lr_init,
                                              trainable=False)
        # report the current learn rate to tf log
        tf.summary.scalar('learn_rate', self.learn_rate_var)

      with tf.variable_scope("trainstep"):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate_var,
                                                beta1=self.TRAIN["decay1"],
                                                beta2=self.TRAIN["decay2"],
                                                epsilon=self.TRAIN["epsilon"])

      # inititialize inference graph
      self.logits_train_list = []
      self.logits_valid_list = []
      self.losses = []
      self.tower_grads = []
      self.img_pl_list = []
      self.lbls_pl_list = []
      self.confusion_matrixes = []

      print("Initializing network")
      with tf.name_scope("train_model"):
        with tf.variable_scope("model"):
          for i in range(self.n_gpus):
            with tf.device(self.assign_to_device('/gpu:%d' % i)):
              print(' TRAINING GRAPH '.center(80, '*'))
              print(' GRAPH GPU:%d ' % i)
              # get placeholders
              img_pl, lbls_pl = self.placeholders(
                  self.DATA["img_prop"]["depth"], self.batch_size_gpu)
              self.img_pl_list.append(img_pl)
              self.lbls_pl_list.append(lbls_pl)

              # graph
              logits_train, _, _ = self.build_graph(img_pl, True)  # train
              self.logits_train_list.append(logits_train)

              # define the loss function, and calculate the gradients
              with tf.name_scope("loss_%d" % i):
                loss = self.loss_f(lbls_pl, logits_train, self.TRAIN["gamma"],
                                   self.TRAIN["loss"], self.TRAIN["w_decay"])
                if self.TRAIN["grads"] == "speed":
                  # calculate tower grads by using OpenAI's implementation
                  # of checkpointed gradients (better for memory)
                  grads = msg.gradients_speed(
                      loss, tf.trainable_variables(), gate_gradients=True)
                elif self.TRAIN["grads"] == "mem":
                  # calculate tower grads by using OpenAI's implementation
                  # of checkpointed gradients (better for speed)
                  grads = msg.gradients_memory(
                      loss, tf.trainable_variables(), gate_gradients=True)
                elif self.TRAIN["grads"] == "tf":
                  # calculate tower grads by using TF implementation
                  print("Using tensorflow gradients")
                  grads = tf.gradients(
                      loss, tf.trainable_variables())
                else:
                  print("Gradient option not supported. Check config")
                grads_and_vars = list(zip(grads, tf.trainable_variables()))

              # append to the list of gradients and losses
              self.losses.append(loss)
              self.tower_grads.append(grads_and_vars)

              # Reuse variables for the next tower.
              tf.get_variable_scope().reuse_variables()

              print('*' * 80)
      with tf.name_scope("test_model"):
        with tf.variable_scope("model", reuse=True):
          for i in range(self.n_gpus):
            with tf.device('/gpu:%d' % i):
              print(' TESTING GRAPH '.center(80, '*'))
              print(' GRAPH GPU:%d ' % i)
              img_pl = self.img_pl_list[i]
              lbls_pl = self.lbls_pl_list[i]
              logits_valid, _, _ = self.build_graph(img_pl, False)  # test
              self.logits_valid_list.append(logits_valid)

              # create a confusion matrix to run with every training step
              with tf.variable_scope("confusion"):
                lbls_resized = self.resize_label(lbls_pl)
                lbls_flattened = tf.reshape(lbls_resized, [-1])
                argmax_flattened = tf.reshape(
                    tf.argmax(logits_valid, axis=3), [-1])
                conf_mat = tf.confusion_matrix(argmax_flattened,
                                               lbls_flattened,
                                               num_classes=self.num_classes,
                                               dtype=tf.float32,
                                               name="confusion_matrix")
                self.confusion_matrixes.append(conf_mat)

              print('*' * 80)

      # print number of parameters (just a check)
      n_parameters = 0
      for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'):
        var_params = np.prod(var.get_shape().as_list())
        n_parameters += var_params
      print("Total number of parameters in network: ", n_parameters)

      with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        self.grads = self.average_gradients(self.tower_grads)

        # total loss
        self.loss = tf.add_n(self.losses)

        # confusion matrix and total logits
        self.confusion_matrix = tf.add_n(self.confusion_matrixes)
        self.logits_valid = tf.concat(self.logits_valid_list, 0)

        # Add histograms for gradients.
        if self.TRAIN['summary']:
          for grad, var in self.grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          self.train_op = self.optimizer.apply_gradients(self.grads)

      # define the best performance so far in the validation set
      self.best_acc_validation = 0
      self.best_iou_validation = 0

      # periodically report accuracy and IoU
      # accuracy
      print("Reporting accuracy every ",
            self.TRAIN["acc_report_epochs"], " epochs")
      with tf.variable_scope("accuracies"):
        self.train_accuracy = tf.Variable(0.0, name="train", trainable=False)
        self.validation_accuracy = tf.Variable(
            0.0, name="validation", trainable=False)
        self.test_accuracy = tf.Variable(0.0, name="test", trainable=False)
        # summaries for the accuracies (to be evaluated later on)
        tf.summary.scalar('train_accuracy', self.train_accuracy)
        tf.summary.scalar('validation_accuracy', self.validation_accuracy)
        tf.summary.scalar('test_accuracy', self.test_accuracy)
      # IoU
      with tf.variable_scope("IoU"):
        self.train_IoU = tf.Variable(0.0, name="train", trainable=False)
        self.validation_IoU = tf.Variable(
            0.0, name="validation", trainable=False)
        self.test_IoU = tf.Variable(0.0, name="test", trainable=False)
        # summaries for the accuracies (to be evaluated later on)
        tf.summary.scalar('train_IoU', self.train_IoU)
        tf.summary.scalar('validation_IoU', self.validation_IoU)
        tf.summary.scalar('test_IoU', self.test_IoU)
      with tf.variable_scope("loss_value"):
        # Add a scalar summary for the snapshot loss.
        self.train_loss = tf.Variable(0.0, name="train_loss", trainable=False)
        tf.summary.scalar('train', self.train_loss)
        self.train_batch_iou = tf.Variable(
            0.0, name="train_batch_iou", trainable=False)
        tf.summary.scalar('batch_iou', self.train_batch_iou)
        self.train_batch_acc = tf.Variable(
            0.0, name="train_batch_acc", trainable=False)
        tf.summary.scalar('batch_acc', self.train_batch_acc)

      # Build the summary Tensor based on the TF collection of Summaries.
      self.summary = tf.summary.merge_all()

      # Add the variable initializer Op.
      self.init = tf.global_variables_initializer()

      # Create a saver for writing training checkpoints.
      self.saver = tf.train.Saver(save_relative_paths=True)

      # xla stuff for faster inference (and soft placement for low ram device)
      gpu_options = tf.GPUOptions(allow_growth=True, force_gpu_compatible=True)
      config = tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False, gpu_options=gpu_options)
      config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

      # start a session
      self.sess = tf.Session(config=config)

      # Instantiate a SummaryWriter to output summaries and the Graph.
      self.log_dir = self.log + '/lr_' + str(self.lrate)
      print("Saving this iteration of training in %s" % self.log_dir)
      self.summary_writer = tf.summary.FileWriter(
          self.log_dir, self.sess.graph)

      # Run the Op to initialize the variables.
      self.sess.run(self.init)

      # if path to model is give, try to restore:
      if path is not None:
        self.restore_session(path)

      # do the training
      print("Training model")

      # Start the training loop
      steps_per_epoch = self.dataset.train.num_examples / \
          float(self.batch_size)
      acc_report_steps = int(self.TRAIN["acc_report_epochs"] * steps_per_epoch)
      max_steps = int(self.TRAIN["max_epochs"] * steps_per_epoch)
      print("Training network %d epochs (%d iterations at batch size %d)" %
            (self.TRAIN["max_epochs"], max_steps, self.TRAIN["batch_size"]))

      # calculate the decay steps with the batch size and num examples
      self.decay_steps = int(self.TRAIN["lr_decay"] * steps_per_epoch)
      self.decay_rate = float(self.TRAIN["lr_rate"])
      print("Decaying learn rate by %f every %d epochs (%d steps)" %
            (self.decay_rate, self.TRAIN["lr_decay"], self.decay_steps))
      for self.step in range(max_steps):
        # do learn rate decay
        if self.step % self.decay_steps == 0 and self.step > 0:
          assign_lr = self.learn_rate_var.assign(
              self.sess.run(self.learn_rate_var) / self.decay_rate)
          self.sess.run(assign_lr)  # assign the value to the node
          print("Decreased learning rate to: ",
                self.sess.run(self.learn_rate_var))

        # fill in the dictionaries
        start_time = time.time()
        feed_dict, _ = self.fill_feed_dict(self.dataset.train,
                                           self.img_pl_list,
                                           self.lbls_pl_list,
                                           self.batch_size_gpu)
        duration_get_batch = time.time() - start_time

        # Run one step of the model in all gpus
        start_time = time.time()
        _, self.loss_value = self.sess.run(
            [self.train_op, self.loss], feed_dict=feed_dict)
        duration_train_step = time.time() - start_time

        # Print status to stdout.
        print('Epoch: %d. Step %d: loss = %.5f (train step: %.3f sec, get_batch: %.3f)'
              % (self.step / steps_per_epoch, self.step, self.loss_value,
                 duration_train_step, duration_get_batch))
        # Write the summaries
        if self.step % self.TRAIN["summary_freq"] == 0:
          # write loss summary
          train_loss_op = self.train_loss.assign(self.loss_value)
          self.sess.run(train_loss_op)

          # write batch iou summary
          pix_h = self.sess.run(self.confusion_matrix, feed_dict=feed_dict)
          if self.TRAIN["ignore_crap"]:
            pix_h = pix_h[:-1, :-1]
          mean_acc, mean_iou, _, _, _ = self.pix_acc_from_histogram(pix_h)
          self.sess.run(self.train_batch_iou.assign(mean_iou))
          self.sess.run(self.train_batch_acc.assign(mean_acc))

          # Update the events file only if I wont do that later on
          print("Saving summaries")
          self.summary_str = self.sess.run(self.summary, feed_dict=feed_dict)
          # add_summary takes ints, so x axis in log will be epoch * 1000
          fake_epoch = int(self.step / float(steps_per_epoch) * 1000)
          self.summary_writer.add_summary(self.summary_str, fake_epoch)
          self.summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if self.step % acc_report_steps == 0 or (self.step + 1) == max_steps:
          # Evaluate against the training set.
          print('Training Data Eval:')
          ignore_last = self.TRAIN["ignore_crap"]
          m_acc, m_iou, _, _, _ = self.training_dataset_accuracy(
              self.dataset.train, self.batch_size, self.batch_size_gpu, ignore_last)
          acc_op = self.train_accuracy.assign(m_acc)
          iou_op = self.train_IoU.assign(m_iou)
          self.sess.run([acc_op, iou_op])  # assign the value to the nodes

          # Evaluate against the validation set.
          print('Validation Data Eval:')
          m_acc, m_iou, _, _, _ = self.training_dataset_accuracy(
              self.dataset.validation, self.batch_size, self.batch_size_gpu, ignore_last)
          acc_op = self.validation_accuracy.assign(m_acc)
          iou_op = self.validation_IoU.assign(m_iou)
          self.sess.run([acc_op, iou_op])  # assign the value to the nodes

          # if the validation performance is the best yet, replace saved model
          if m_acc > self.best_acc_validation:
            acc_log_folder = self.log + "/acc/"
            if not tf.gfile.Exists(acc_log_folder):
              tf.gfile.MakeDirs(acc_log_folder)
            # save a checkpoint
            self.best_acc_validation = m_acc
            self.acc_checkpoint_file = os.path.join(
                acc_log_folder, 'model-best-acc.ckpt')
            self.saver.save(self.sess, self.acc_checkpoint_file)
            # report to user
            print("Best validation accuracy yet, saving network checkpoint")

          if m_iou > self.best_iou_validation:
            iou_log_folder = self.log + "/iou/"
            if not tf.gfile.Exists(iou_log_folder):
              tf.gfile.MakeDirs(iou_log_folder)
            # save a checkpoint
            self.best_iou_validation = m_iou
            self.iou_checkpoint_file = os.path.join(
                iou_log_folder, 'model-best-iou.ckpt')
            self.saver.save(self.sess, self.iou_checkpoint_file)
            # report to user
            print("Best validation mean IoU yet, saving network checkpoint")

          # Evaluate against the test set.
          print('Test Data Eval:')
          m_acc, m_iou, _, _, _ = self.training_dataset_accuracy(
              self.dataset.test, self.batch_size, self.batch_size_gpu, ignore_last)
          acc_op = self.test_accuracy.assign(m_acc)
          iou_op = self.test_IoU.assign(m_iou)
          self.sess.run([acc_op, iou_op])  # assign the value to the nodes

          # summarize
          self.summary_str = self.sess.run(self.summary, feed_dict=feed_dict)
          # add_summary takes ints, so x axis in log will be epoch * 1000
          fake_epoch = int(self.step / float(steps_per_epoch) * 1000)
          self.summary_writer.add_summary(self.summary_str, fake_epoch)
          self.summary_writer.flush()

  def placeholders(self, depth, batch_size):
    """Generate placeholder variables to represent the input tensors
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Return:
      img_pl: placeholder for inputs
      lbls_pl: placeholder for labels
    """
    img_pl = tf.placeholder(tf.float32, shape=(
        batch_size, None, None, depth), name="x_pl")
    lbls_pl = tf.placeholder(tf.int32, shape=(
        batch_size, None, None), name="y_pl")
    return img_pl, lbls_pl

  def fill_feed_dict(self, data_set, img_pl_list, lbls_pl_list, batch_size):
    """Fills the feed_dict for training the given step.
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      img_pl: Placeholder list for images (one item per gpu)
      lbls_pl: Placeholder list for labels (one item per gpu)
      batch_size: Batch size for getting dataset batch
    Returns:
      feed_dict: to be fed to training op (or cnn forward pass)
      name_list:  names of files in batch
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    name_list = []
    feed_dict = {}

    for i in range(0, len(img_pl_list)):
      images_feed, labels_feed, names = data_set.next_batch(batch_size)
      feed_dict[img_pl_list[i]] = images_feed
      feed_dict[lbls_pl_list[i]] = labels_feed
      name_list.append(names)
    return feed_dict, name_list

  def cleanup(self, signum, frame):
    print('Killing all threads and exiting!')
    if hasattr(self, 'dataset') and self.dataset is not None:
      self.dataset.cleanup()
    sys.exit(0)
