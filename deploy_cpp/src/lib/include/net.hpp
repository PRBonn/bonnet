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

// standard stuff
#include <string>
#include <vector>

// opencv
#include <opencv2/core/core.hpp>

// yamlcpp
#include "yaml-cpp/yaml.h"

namespace bonnet {

enum retCode { CNN_OK = 0, CNN_FAIL = 1, CNN_FATAL = 2, CNN_UNDEF = 3 };

/**
 * @brief      Class for network inference.
 */
class Net {
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
  Net(const std::string& model_path, const YAML::Node& cfg_train,
      const YAML::Node& cfg_net, const YAML::Node& cfg_data,
      const YAML::Node& cfg_nodes);

  /**
   * @brief      Destroys the object.
   */
  virtual ~Net();

  /**
   * @brief      Initializes the object
   *
   * @param[in]  device          The device to run the graph (GPU, CPU, TPU
   * **BAZZINGA**)
   * @param[in]  mem_percentage  The memory percentage (0 to 1, of GPU memory)
   *
   * @return     Exit code
   */
  virtual retCode init(const std::string& device,
                       const float& mem_percentage = 1) {
    return CNN_UNDEF;
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
  virtual retCode infer(const cv::Mat& image, cv::Mat& mask,
                        const bool verbose = false) {
    return CNN_UNDEF;
  }

  /**
   * @brief      Generate dictionaries from yaml files
   *
   * @return     Exit code
   */
  retCode init_color();

  /**
   * @brief      Convert mask to color using dictionary
   *
   * @param[in]  mask        The mask from argmax
   * @param[out] color_mask  The output color mask
   * @param[in]  verbose     Verbose output? (such as time to run)
   *
   * @return     Exit code
   */
  retCode color(const cv::Mat& mask, cv::Mat& color_mask,
                const bool verbose = false);

  /**
   * @brief      Blend image with color mask
   *
   * @param[in]  img         Image being inferred
   * @param[in]  alpha       Constant for image
   * @param[in]  color_mask  Color mask from CNN
   * @param[in]  beta        Constant for color mask
   * @param[out] blend       Output blend
   * @param[in]  verbose     Verbose output? (such as time to run)
   *
   * @return     Exit code
   */
  retCode blend(const cv::Mat& img, const float& alpha,
                const cv::Mat& color_mask, const float& beta,
                cv::Mat& blend, const bool verbose = false);

  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  virtual retCode verbosity(const bool verbose) { return CNN_UNDEF; }

 protected:
  // network stuff
  std::string _dev;  // device for the graph "\cpu:0" or "\cpu:0"
  std::string _model_path;

  // config files
  YAML::Node _cfg_train;
  YAML::Node _cfg_net;
  YAML::Node _cfg_data;
  YAML::Node _cfg_nodes;
  std::vector<cv::Vec3b> _argmax_to_bgr;  // for color conversion
};

} /* namespace bonnet */
