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

// to work with images
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// network library (conditional build)
#ifdef TF_AVAIL
#include <netTF.hpp>
#endif  // TF_AVAIL
#ifdef TRT_AVAIL
#include <netTRT.hpp>
#endif  // TRT_AVAIL
#if !defined TF_AVAIL && !defined TRT_AVAIL
#error("At least TF OR TensorRT must be installed")
#endif

// yamlcpp
#include "yaml-cpp/yaml.h"

// for exceptions in constructor
#include <stdexcept>

namespace bonnet {

/**
 * @brief      Handler class for network inference.
 */
class Bonnet {
 public:
  /**
   * @brief      Constructs the global handler
   *
   * @param[in]  path     The path to frozen model directory
   * @param[in]  backend  The backend (tf or trt)
   * @param[in]  dev      The device to infer (/gpu:0, /cpu:0)
   * @param[in]  verbose  Boolean verbose or not
   */
  Bonnet(const std::string& path, const std::string backend,
         const std::string& dev, const bool verbose = false);

  /**
   * @brief      Destroys the object.
   */
  virtual ~Bonnet();

  /**
   * @brief      Infer mask from image
   *
   * @param[in]  image    The image to process
   * @param[out] mask     The mask output as argmax of probabilities
   * @param[in]  verbose  Verbose mode (Output timing)
   *
   * @return     Exit code
   */
  retCode infer(const cv::Mat& image, cv::Mat& mask,
                const bool verbose = false);

  /**
   * @brief      Color a mask according to data config map.
   *
   * @param[in]  mask     The mask
   * @param      color    The colored mask
   * @param[in]  verbose  Verbose mode?
   *
   * @return     Exit code
   */
  retCode color(const cv::Mat& mask, cv::Mat& color,
                const bool verbose = false);

  /**
   * @brief      Blend image with color mask
   *
   * @param[in]  img         Image being inferred
   * @param[in]  alpha       Constant for image
   * @param[in]  mask        Color mask from CNN
   * @param[in]  beta        Constant for color mask
   * @param[out] blend       Output blend
   * @param[in]  verbose     Verbose output? (such as time to run)
   *
   * @return     Exit code
   */
  retCode blend(const cv::Mat& img, const float& alpha, const cv::Mat& mask,
                const float& beta, cv::Mat& blend, const bool verbose);

 private:
  std::string _path;     // path to frozen directory
  std::string _backend;  // backend to use (tf = TensorFlow, trt = TensorRT)
  std::string _device;   // device to infer in (/gpu:0, /cpu:0)
  bool _verbose;         // verbose?
  std::string _model;    // .pb file to pick up

  // define dictionaries for net
  YAML::Node _cfg_train;
  YAML::Node _cfg_net;
  YAML::Node _cfg_data;
  YAML::Node _cfg_nodes;

  // network unique pointer
  std::unique_ptr<bonnet::Net> _net;
};

} /* namespace bonnet */
