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
#pragma once

// tensorflow
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"

// abstract net
#include <net.hpp>

namespace tf = tensorflow;

namespace bonnet {

/**
 * @brief      Class for Tensorflow network inference.
 */
class NetTF : public Net {
 public:
  /**
   * @brief      Constructs the object.
   *
   * @param[in]  model_path  The model path for the frozen pb
   * @param[in]  cfg_train   The configuration file when trained
   * @param[in]  cfg_net     The configuration for network
   * @param[in]  cfg_data    The configuration for the dataset
   * @param[in]  cfg_nodes   The inference nodes
   */
  NetTF(const std::string& model_path, const YAML::Node& cfg_train,
        const YAML::Node& cfg_net, const YAML::Node& cfg_data,
        const YAML::Node& cfg_nodes);

  /**
   * @brief      Destroys the object.
   */
  virtual ~NetTF();

  /**
     * @brief      Initializes the object
     *
     * @param[in]  device          The device to run the graph (GPU, CPU, TPU
     * **BAZZINGA**)
     * @param[in]  mem_percentage  The memory percentage (0 to 1, of GPU memory)
     *
     * @return     Exit code
     */
  retCode init(const std::string& device, const float& mem_percentage = 1);

  /**
   * @brief      Infer mask from image
   *
   * @param[in]  image    The image to process
   * @param[out]  mask     The mask output as argmax of probabilities
   * @param[in]  verbose  Verbose mode (Output timing)
   *
   * @return     Exit code
   */
  retCode infer(const cv::Mat& image, cv::Mat& mask,
                const bool verbose = false);
  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  retCode verbosity(const bool verbose);

 private:
  // tensorflow stuff
  tf::GraphDef _graph_def;    // graph from frozen protobuf
  tf::SessionOptions _opts;   // gpu options
  tf::Session* _session = 0;  // session to run the graph in tf back end
  tf::Status _status;         // status check for each tf action trial

  // graph nodes for i/o
  std::string _input_node;   // name of input node in tf graph
  std::string _output_node;  // name of output node in tf graph
};

} /* namespace bonnet */
