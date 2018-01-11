/* Copyright 2017 Andres Milioto. All Rights Reserved.
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

// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// c++ stuff
#include <chrono>
#include <iomanip>  // for setfill
#include <iostream>
#include <string>

// network library
#ifdef TF_AVAIL
#include <netTF.hpp>
#endif  // TF_AVAIL
#ifdef TRT_AVAIL
#include <netTRT.hpp>
#endif  // TRT_AVAIL
#if (!defined TF_AVAIL) && (!defined TRT_AVAIL)
#error("At least TF OR TensorRT must be installed")
#endif

// boost
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

// yamlcpp
#include "yaml-cpp/yaml.h"

int main(int argc, const char *argv[]) {
  // define options
  std::string video = "";
  std::string log = "/tmp/net_predict_log_video_cpp/";
  std::string path;
  std::string model = "frozen_nchw";
  bool verbose = false;
  std::string device = "/gpu:0";
  std::string backend = "tf";

  // Parse options
  try {
    po::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")(
        "video,v", po::value<std::string>(),
        "Video to infer. Defaults to webcam.")(
        "log,l", po::value<std::string>(),
        "Directory to log output of predictions.")(
        "path,p", po::value<std::string>(),
        "Directory to get the frozen pb model. No default")(
        "model,m", po::value<std::string>(),
        "Type of model (frozen_nchw, frozen_nhwc, optimized, or quantized)")(
        "verbose", po::bool_switch(),
        "Verbose mode. Calculates profile (time to run)")(
        "dev,d", po::value<std::string>(),
        "Device to run on. Example (/gpu:0 or /cpu:0)")(
        "backend,b", po::value<std::string>(),
        "Backend. Tensorflow and TensorRT for now. Later MovidiusNCSDK.");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }

    if (vm.count("video")) {
      video = vm["video"].as<std::string>();
      std::cout << "video: " << video << std::endl;
    } else {
      std::cout << "video: Using default (camera)!" << std::endl;
    }

    if (vm.count("log")) {
      log = vm["log"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "log: " << log << std::endl;
    } else {
      std::cout << "log: " << log << ". Using default!" << std::endl;
    }
    if (vm.count("path")) {
      path = vm["path"].as<std::string>() + "/";  // make sure path is valid
      std::cout << "path: " << path << std::endl;
    } else {
      std::cerr << "No path! See --help (-h) for help. Exiting" << std::endl;
      return 1;
    }
    if (vm.count("model")) {
      model = vm["model"].as<std::string>();
      std::cout << "model: " << model << std::endl;
    } else {
      std::cout << "model: " << model << ". Using default!" << std::endl;
    }
    if (vm.count("verbose")) {
      verbose = vm["verbose"].as<bool>();
      std::cout << "verbose: " << verbose << std::endl;
    } else {
      std::cout << "verbose: " << verbose << ". Using default!" << std::endl;
    }
    if (vm.count("dev")) {
      device = vm["dev"].as<std::string>();
      std::cout << "device: " << device << std::endl;
    } else {
      std::cout << "device: " << device << ". Using default!" << std::endl;
    }
    if (vm.count("backend")) {
      backend = vm["backend"].as<std::string>();
      std::cout << "backend: " << backend << std::endl;
    } else {
      std::cout << "backend: " << backend << ". Using default!" << std::endl;
    }

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  } catch (const po::error &ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  // create the log folder
  try {
    if (fs::exists(log)) {
      fs::remove_all(log);
    }
    fs::create_directory(log);
  } catch (fs::filesystem_error const &e) {
    std::cerr << "Failed to create the log directory: " << log << std::endl;
    std::cerr << e.what();
    return 1;
  }
  std::cout << "Successfully created log directory: " << log << std::endl;

  // define dictionaries for net
  YAML::Node cfg_train;
  YAML::Node cfg_net;
  YAML::Node cfg_data;
  YAML::Node cfg_nodes;
  try {
    cfg_train = YAML::LoadFile(path + "train.yaml");
    cfg_net = YAML::LoadFile(path + "net.yaml");
    cfg_data = YAML::LoadFile(path + "data.yaml");
    cfg_nodes = YAML::LoadFile(path + "nodes.yaml");
  } catch (YAML::Exception &ex) {
    std::cerr << "Can't open one of the config files from the model path "
                 "directory. Exception: "
              << ex.what() << std::endl;
    return 1;
  }

  // network stuff
  cv::Mat mask_argmax;
  bonnet::retCode status;
  std::unique_ptr<bonnet::Net> net;
  if (backend == "tf") {
#ifdef TF_AVAIL
    // form the model path from path and model
    std::string full_model_path = path + model + ".pb";
    std::cout << "Full model path: " << full_model_path << std::endl;

    // generate net with tf backend
    net = std::unique_ptr<bonnet::Net>(new bonnet::NetTF(
        full_model_path, cfg_train, cfg_net, cfg_data, cfg_nodes));

    // set verbosity
    net->verbosity(verbose);
#else
    std::cerr << "Tensorflow is supported, but build was not able to find it."
              << std::endl;
    return 1;
#endif  // TF_AVAIL
  } else if (backend == "trt") {
#ifdef TRT_AVAIL
    // form the model path from path and model
    std::string full_model_path = path + "optimized_tRT.uff";
    std::cout << "Full model path: " << full_model_path << std::endl;

    // generate net with TensorRT backend
    net = std::unique_ptr<bonnet::Net>(new bonnet::NetTRT(
        full_model_path, cfg_train, cfg_net, cfg_data, cfg_nodes));
    // set verbosity
    net->verbosity(verbose);
#else
    std::cerr << "TensorRT is supported, but build was not able to find it."
              << std::endl;
    return 1;
#endif  // TRT_AVAIL
  } else {
    std::cerr << "Tensorflow (tf) and Tensor RT (trt) backends are the only"
                 "ones implemented right now. Exiting..."
              << std::endl;
    return 1;
  }

  // initialize network
  status = net->init(device);
  if (status != bonnet::CNN_OK) {
    std::cerr << "Failed to intialize CNN." << std::endl;
    return 1;
  }

  // open capture
  std::unique_ptr<cv::VideoCapture> cap;
  if (video == "") {
    std::cout << "Opening webcam for prediction." << std::endl;
    cap = std::unique_ptr<cv::VideoCapture>(new cv::VideoCapture(0));
  } else {
    std::cout << "Opening video" << video << " for prediction." << std::endl;
    cap = std::unique_ptr<cv::VideoCapture>(new cv::VideoCapture(video));
  }
  if (!cap->isOpened())  // check if we succeeded
  {
    return 1;
  }

  // predict each frame
  for (int i = 0;; i++) {
    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
    std::cout << "Predicting frame: " << i << std::endl;

    // Open a frame
    cv::Mat frame;
    *cap >> frame;  // get a new frame from camera
    // Check for invalid input
    if (!frame.data) {
      std::cerr << "No image in frame!" << std::endl;
      return 1;
    }

    // predict
    status = net->infer(frame, mask_argmax, verbose);
    if (status != bonnet::CNN_OK) {
      std::cerr << "Failed to run CNN." << std::endl;
      return 1;
    }

    // convert xentropy mask to colors using dictionary
    cv::Mat mask_bgr(mask_argmax.rows, mask_argmax.cols, CV_8UC3);
    status = net->color(mask_argmax, mask_bgr, verbose);
    if (status != bonnet::CNN_OK) {
      std::cerr << "Failed to color result of CNN." << std::endl;
      return 1;
    }

    // save image to log directory
    std::string image_log_path =
        log + fs::path(std::to_string(i)).stem().string() + ".png";
    std::cout << "Saving this image to " << image_log_path << std::endl;
    cv::imwrite(image_log_path, mask_bgr);

    // print the output
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", mask_bgr);  // Show the frame
    if (cv::waitKey(1) == 27) {
      break;
    };

    std::cout << std::setfill('=') << std::setw(80) << "" << std::endl;
  }

  return 0;
}
