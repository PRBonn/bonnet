/* Copyright 2017 Andres Milioto, Cyrill Stachniss. All Rights Reserved.
 *
 *  This file is part of Bonnet.
 *
 *  Bonnet is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Bonnet is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Bonnet. If not, see <http://www.gnu.org/licenses/>.
 */

// own definition
#include <netTF.hpp>

// standard stuff
#include <stdlib.h>
#include <chrono>
#include <iostream>

namespace bonnet {

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the frozen pb
 * @param[in]  cfg_train   The configuration file when trained
 * @param[in]  cfg_net     The configuration for network
 * @param[in]  cfg_data    The configuration for the dataset
 * @param[in]  cfg_nodes   The inference nodes
 */
NetTF::NetTF(const std::string& model_path, const YAML::Node& cfg_train,
             const YAML::Node& cfg_net, const YAML::Node& cfg_data,
             const YAML::Node& cfg_nodes)
    : Net(model_path, cfg_train, cfg_net, cfg_data, cfg_nodes) {}

/**
 * @brief      Destroys the object.
 */
NetTF::~NetTF() {
  // Free any resources used by the session
  std::cout << "Closing session and exiting." << std::endl;
  if (_session) {
    _status = _session->Close();
    if (!_status.ok()) {
      std::cerr << _status.ToString() << std::endl;
    }
  }
}

/**
 * @brief      Initializes the object
 *
 * @param[in]  device          The device to run the graph (GPU, CPU, TPU
 * **BAZZINGA**)
 * @param[in]  mem_percentage  The memory percentage (0 to 1, of GPU memory)
 *
 * @return     Exit code
 */
retCode NetTF::init(const std::string& device, const float& mem_percentage) {
  // get the device
  if (device == "/gpu:0" || device == "/cpu:0") {
    _dev = device;
  } else {
    std::cerr << "DEVICE " << device << " doesn't make sense yo!" << std::endl;
    return CNN_FATAL;
  }

  // get the nodes in the tf graph to run (i/o)
  try {
    _input_node = _cfg_nodes["input_node"].as<std::string>();
    _output_node = _cfg_nodes["mask_node"].as<std::string>();
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one of the node names from the nodes.yaml file"
              << ex.what() << std::endl;
    return CNN_FATAL;
  }

  // Read in the protobuf graph we exported
  _status = ReadBinaryProto(tf::Env::Default(), _model_path, &_graph_def);
  if (!_status.ok()) {
    std::cerr << _status.ToString() << std::endl;
    return CNN_FATAL;
  }
  std::cout << "Successfully imported frozen protobuf." << std::endl;

  // Set options for session
  tf::graph::SetDefaultDevice(_dev, &_graph_def);
  _opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
      mem_percentage);
  _opts.config.mutable_gpu_options()->set_allow_growth(true);
  _opts.config.set_allow_soft_placement(true);

  // Start a session
  _status = tf::NewSession(_opts, &_session);
  if (!_status.ok()) {
    std::cerr << _status.ToString() << std::endl;
    return CNN_FATAL;
  }
  std::cout << "Session successfully created.\n";

  // Add the graph to the session
  _status = _session->Create(_graph_def);
  if (!_status.ok()) {
    std::cerr << _status.ToString() << std::endl;
    return CNN_FATAL;
  }
  std::cout << "Successfully added graph to session." << std::endl;

  // get the proper color dictionary
  if (init_color() != CNN_OK) {
    return CNN_FAIL;
  }

  return CNN_OK;
}

/**
 * @brief      Infer mask from image
 *
 * @param[in]  image    The image to process
 * @param[out] mask     The mask output as argmax of probabilities
 * @param[in]  verbose  Verbose mode (Output timing)
 *
 * @return     Exit code
 */
retCode NetTF::infer(const cv::Mat& image, cv::Mat& mask, const bool verbose) {
  // Check if image has something
  if (!image.data) {
    std::cout << "Could find content in the image" << std::endl;
    return CNN_FAIL;
  }

  // get the start time to report
  auto start_total = std::chrono::high_resolution_clock::now();

  // Get dimensions
  unsigned int cv_img_h = image.rows;
  unsigned int cv_img_w = image.cols;
  unsigned int cv_img_d = image.channels();

  // Check that it has 3 channels
  unsigned int expected_depth =
      _cfg_data["img_prop"]["depth"].as<unsigned int>();
  assert(cv_img_d == expected_depth);

  // Set up inputs to run the graph
  // tf tensor for feeding the graph
  tf::Tensor x_pl(tf::DT_FLOAT, {1, cv_img_h, cv_img_w, cv_img_d});

  // tf pointer for init of fake cv mat
  float* x_pl_pointer = x_pl.flat<float>().data();

  // fake cv mat (avoid copy)
  cv::Mat x_pl_cv(cv_img_h, cv_img_w, CV_32FC3, x_pl_pointer);
  image.convertTo(x_pl_cv, CV_32FC3);

  // feed the input
  std::vector<std::pair<std::string, tf::Tensor>> inputs = {
      {_input_node, x_pl}};

  // The session will initialize the outputs automatically
  std::vector<tf::Tensor> outputs;

  // Run the session, evaluating our all operation from the graph
  auto start_inference = std::chrono::high_resolution_clock::now();
  _status = _session->Run(inputs, {_output_node}, {}, &outputs);
  if (!_status.ok()) {
    std::cerr << _status.ToString() << std::endl;
    return CNN_FATAL;
  }
  auto elapsed_inference =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start_inference)
          .count();

  // Process the output with map
  // Get output dimensions
  unsigned int output_img_n = outputs[0].shape().dim_size(0);
  unsigned int output_img_h = outputs[0].shape().dim_size(1);
  unsigned int output_img_w = outputs[0].shape().dim_size(2);
  unsigned int output_img_c = outputs[0].shape().dim_size(3);
  if (verbose) {
    std::cout << "shape of output: h:" << output_img_h << ",w:" << output_img_w
              << ",c:" << output_img_c << ",n:" << output_img_n << std::endl;
  }
  // tf pointer for init of fake cv mat
  int32_t* output_pointer = outputs[0].flat<int32_t>().data();

  // fake cv mat (avoid copy)
  cv::Mat output_cv(output_img_h, output_img_w, CV_32S, output_pointer);
  cv::Mat output_cv_8b(output_img_h, output_img_w, CV_8U);
  output_cv.convertTo(mask, CV_8U);

  auto elapsed_total =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start_total)
          .count();

  if (verbose) {
    std::cout << "Successfully run prediction from session." << std::endl;
    std::cout << "Time to infer: " << elapsed_inference << "ms." << std::endl;
    std::cout << "Time in total: " << elapsed_total << "ms." << std::endl;
  }

  return CNN_OK;
}

/**
 * @brief      Set verbosity level for backend execution
 *
 * @param[in]  verbose  True is max verbosity, False is no verbosity.
 *
 * @return     Exit code.
 */
retCode NetTF::verbosity(const bool verbose) {
  if (verbose) {
    setenv("TF_CPP_MIN_LOG_LEVEL", "0", true);
  } else {
    setenv("TF_CPP_MIN_LOG_LEVEL", "3", true);
  }

  return CNN_OK;
}

} /* namespace bonnet */
