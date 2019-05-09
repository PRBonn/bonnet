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

#include <bonnet.hpp>

namespace bonnet {

/**
   * @brief      Constructs the global handler
   *
   * @param[in]  path     The path to frozen model directory
   * @param[in]  backend  The backend (tf or trt)
   * @param[in]  dev      The device to infer (/gpu:0, /cpu:0)
   * @param[in]  verbose  Boolean verbose or not
   */
Bonnet::Bonnet(const std::string& path, const std::string backend,
               const std::string& dev, const bool verbose) {
  // get the values
  _path = path;
  _backend = backend;
  _device = dev;
  _verbose = verbose;

  // as a first step, try to get the config files
  try {
    _cfg_train = YAML::LoadFile(path + "/train.yaml");
  } catch (YAML::Exception& ex) {
    throw std::invalid_argument("Invalid yaml file " + path + "/train.yaml");
  }

  try {
    _cfg_net = YAML::LoadFile(path + "/net.yaml");
  } catch (YAML::Exception& ex) {
    throw std::invalid_argument("Invalid yaml file " + path + "/net.yaml");
  }

  try {
    _cfg_data = YAML::LoadFile(path + "/data.yaml");
  } catch (YAML::Exception& ex) {
    throw std::invalid_argument("Invalid yaml file " + path + "/data.yaml");
  }

  try {
    _cfg_nodes = YAML::LoadFile(path + "/nodes.yaml");
  } catch (YAML::Exception& ex) {
    throw std::invalid_argument("Invalid yaml file " + path + "/nodes.yaml");
  }

  // Select the model according to backend and device
  if (_backend == "tf") {
    if (_device.find("gpu") != std::string::npos) {
      _model = path + "/frozen_nchw.pb";
    } else if (_device.find("cpu") != std::string::npos) {
      _model = path + "/frozen_nhwc.pb";
    } else {
      throw std::invalid_argument("Invalid device " + _device);
    }
  } else if (_backend == "trt") {
    _model = path + "/optimized_tRT.uff";
  } else {
    throw std::invalid_argument("Invalid backend" + _backend);
  }

  // output model path
  std::cout << "Full model path: " << _model << std::endl;

  // Then construct the network
  if (_backend == "tf") {
#ifdef TF_AVAIL
    // generate net with tf backend
    _net = std::unique_ptr<Net>(
        new NetTF(_model, _cfg_train, _cfg_net, _cfg_data, _cfg_nodes));

    // set verbosity
    _net->verbosity(_verbose);
#else
    throw std::invalid_argument("Backend supported but not built: " + _backend);
#endif  // TF_AVAIL
  } else if (backend == "trt") {
#ifdef TRT_AVAIL
    // generate net with TensorRT backend
    _net = std::unique_ptr<Net>(
        new NetTRT(_model, _cfg_train, _cfg_net, _cfg_data, _cfg_nodes));
    // set verbosity
    _net->verbosity(verbose);
#else
    throw std::invalid_argument("Backend supported but not built: " + _backend);
#endif  // TRT_AVAIL
  } else {
    throw std::invalid_argument("Backend not supported: " + _backend);
  }

  // initialize using device
  retCode status = _net->init(_device);
  if (status != CNN_OK) {
    throw std::runtime_error("Failed to initialize CNN");
  }

  // set verbosity
  _net->verbosity(_verbose);
}

/**
 * @brief      Destroys the object.
 */
Bonnet::~Bonnet() {
  // TODO - Eliminate resources
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
retCode Bonnet::infer(const cv::Mat& image, cv::Mat& mask, const bool verbose) {
  retCode status = _net->infer(image, mask, verbose);
  if (status != CNN_OK) {
    std::cerr << "Failed to run CNN." << std::endl;
    return CNN_FATAL;
  }
  return CNN_OK;
}

/**
   * @brief      Color a mask according to data config map.
   *
   * @param[in]  mask     The mask
   * @param      color    The colored mask
   * @param[in]  verbose  Verbose mode?
   *
   * @return     Exit code
   */
retCode Bonnet::color(const cv::Mat& mask, cv::Mat& color, const bool verbose) {
  retCode status = _net->color(mask, color, verbose);
  if (status != CNN_OK) {
    std::cerr << "Failed to color result of CNN." << std::endl;
    return CNN_FATAL;
  }
  return CNN_OK;
}

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
retCode Bonnet::blend(const cv::Mat& img, const float& alpha,
                      const cv::Mat& mask, const float& beta, cv::Mat& blend,
                      const bool verbose) {
  retCode status = _net->blend(img, alpha, mask, beta, blend, verbose);
  if (status != CNN_OK) {
    std::cerr << "Failed to blend result of CNN." << std::endl;
    return CNN_FATAL;
  }
  return CNN_OK;
}

} /* namespace bonnet */
