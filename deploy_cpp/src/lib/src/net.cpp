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
#include <net.hpp>

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
Net::Net(const std::string& model_path, const YAML::Node& cfg_train,
         const YAML::Node& cfg_net, const YAML::Node& cfg_data,
         const YAML::Node& cfg_nodes) {
  // get the model path
  _model_path = model_path;

  // get the config files
  _cfg_train = cfg_train;
  _cfg_net = cfg_net;
  _cfg_data = cfg_data;
  _cfg_nodes = cfg_nodes;
}

/**
 * @brief      Destroys the object.
 */
Net::~Net() {}

/**
 * @brief      Convert mask to color using dictionary
 *
 * @param[in]  mask        The mask from argmax
 * @param[out] color_mask  The output color mask
 * @param[in]  verbose     Verbose output? (such as time to run)
 *
 * @return     Exit code
 */
retCode Net::color(const cv::Mat& mask, cv::Mat& color_mask,
                   const bool verbose) {
  // "fast" implementation of remapping
  for (int y = 0; y < mask.rows; y++) {
    const uint8_t* M_x = mask.ptr<uint8_t>(y);
    for (int x = 0; x < mask.cols; x++) {
      color_mask.at<cv::Vec3b>(cv::Point(x, y)) = _argmax_to_bgr[M_x[x]];
    }
  }

  return CNN_OK;
}

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
retCode Net::blend(const cv::Mat& img, const float& alpha,
                   const cv::Mat& color_mask, const float& beta, cv::Mat& blend,
                   const bool verbose) {
  cv::addWeighted(img, alpha, color_mask, beta, 0.0, blend);

  return CNN_OK;
}

/**
 * @brief      Generate dictionaries from yaml files
 *
 * @return     Exit code
 */
retCode Net::init_color() {
  // parse the colors for the color conversion
  YAML::Node label_remap;
  YAML::Node color_map;
  try {
    label_remap = _cfg_data["label_remap"];
    color_map = _cfg_data["color_map"];

  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one of the color dictionaries from data.yaml."
              << ex.what() << std::endl;
    return CNN_FATAL;
  }

  // get the remapping from both dictionaries, in order to speed up conversion
  int n_classes = label_remap.size();

  // initialize (easier to do here)
  for (int i = 0; i < n_classes; i++) {
    cv::Vec3b color = {0, 0, 0};      // Create an empty color
    _argmax_to_bgr.push_back(color);  // Add the bgr to the main vector
  }

  YAML::const_iterator it;
  for (it = label_remap.begin(); it != label_remap.end(); ++it) {
    int key = it->first.as<int>();     // <- key
    int label = it->second.as<int>();  // <- argmax label
    cv::Vec3b color = {
        static_cast<uint8_t>(color_map[key][0].as<unsigned int>()),
        static_cast<uint8_t>(color_map[key][1].as<unsigned int>()),
        static_cast<uint8_t>(color_map[key][2].as<unsigned int>())};
    _argmax_to_bgr[label] = color;

    // std::cout << key << ":" << label << std::endl;
    // std::cout << label << ":" << color << std::endl;
  }
  std::cout << "Remapped colors from configuration file." << std::endl;

  return CNN_OK;
}

} /* namespace bonnet */
