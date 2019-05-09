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
#include <netTRT.hpp>

// standard stuff
#include <stdlib.h>
#include <chrono>
#include <iostream>

// opencv
#include "opencv2/opencv.hpp"

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
NetTRT::NetTRT(const std::string& model_path, const YAML::Node& cfg_train,
               const YAML::Node& cfg_net, const YAML::Node& cfg_data,
               const YAML::Node& cfg_nodes)
    : Net(model_path, cfg_train, cfg_net, cfg_data, cfg_nodes) {}

/**
 * @brief      Destroys the object.
 */
NetTRT::~NetTRT() {
  // Free any resources used by the engine
  std::cout << "Closing engine and exiting." << std::endl;
  // TODO - Eliminate resources
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
retCode NetTRT::init(const std::string& device, const float& mem_percentage) {
  // get the device (only one accepted for now, so this if doesn't make sense,
  // but it may in the future)
  if (device == "/gpu:0") {
    _dev = device;
  } else {
    std::cerr << "DEVICE " << device << " doesn't make sense yo!" << std::endl;
    return CNN_FATAL;
  }

  // try to get the input and output node names
  // get the nodes in the tf graph to run (i/o)
  try {
    _input_node = _cfg_nodes["input_norm_and_resized_node"].as<std::string>();
    _output_node = _cfg_nodes["logits_node"].as<std::string>();
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one of the node names from the nodes.yaml file"
              << ex.what() << std::endl;
    return CNN_FATAL;
  }

  // get sizes for cuda malloc
  unsigned int num_classes, d, w, h;
  try {
    num_classes = _cfg_data["label_remap"].size();
    d = _cfg_data["img_prop"]["depth"].as<unsigned int>();
    w = _cfg_data["img_prop"]["width"].as<unsigned int>();
    h = _cfg_data["img_prop"]["height"].as<unsigned int>();
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one of the properties from data.yaml." << ex.what()
              << std::endl;
    return CNN_FATAL;
  }

  // create a builder for inference
  _builder = nvinf::createInferBuilder(_logger);

  // check if there is fp16 support (such as Jetson, or Volta tensor core)
  _fp16 = _builder->platformHasFastFp16();
  std::cout << "Platform " << (_fp16 ? "has" : "doesn't have")
            << " fp16 support!" << std::endl;
  _net_data_type = _fp16 ? nvinf::DataType::kHALF : nvinf::DataType::kFLOAT;

  // create net and parser to parse the uff model from frozen model folder
  _network = _builder->createNetwork();
  _parser = uffpar::createUffParser();

  // register the inputs and outputs
  nvinf::DimsCHW inputDims = {(int)d, (int)h, (int)w};
  _parser->registerInput(_input_node.c_str(), inputDims);
  _parser->registerOutput(_output_node.c_str());

  // parse the model
  auto nodes = _parser->parse(_model_path.c_str(), *_network, _net_data_type);
  if (!nodes) {
    std::cerr << "Can't parse uff model" << std::endl;
    return CNN_FATAL;
  }

  // Build the engine specifying memory and batch properties
  _builder->setMaxBatchSize(1);
  _engine = 0;
  int size = 30;
  do {
    _builder->setMaxWorkspaceSize(1 << size);
    // fp16 engine disabled because it makes TX2 goes bananas in TRT3.
    // I will enable again when I have an answer from NVIDIA
    // if (_fp16) _builder->setHalf2Mode(true);
    _engine = _builder->buildCudaEngine(*_network);
    if (!_engine) {
      std::cout << "Failed to create CUDA engine. Trying smaller workspace size"
                << std::endl;
      size--;
    }
  } while (!_engine && size >= 5);
  if (!_engine) {
    std::cout << "Tried my best, but your GPU sucks right now." << std::endl;
    return CNN_FATAL;
  }

  // create an execution context
  _context = _engine->createExecutionContext();
  if (!_context) {
    std::cout << "Failed to create context for engine excecution." << std::endl;
    return CNN_FATAL;
  }

  // Get the bindings for input and output
  _inputIndex = _engine->getBindingIndex(_input_node.c_str());
  _inputDims = _engine->getBindingDimensions(_inputIndex);
  _outputIndex = _engine->getBindingIndex(_output_node.c_str());
  _outputDims = _engine->getBindingDimensions(_outputIndex);

  // sizes for cuda malloc
  _size_in_pix = w * h;
  _sizeof_in = d * _size_in_pix * sizeof(float);
  _sizeof_out = num_classes * _size_in_pix * sizeof(int);

  // TODO: Check the size of the bindings with the real values, to check for
  // inconsistencies

  // Allocate GPU memory for I/O
  _cuda_buffers[_inputIndex] = &_input_gpu;
  _cuda_buffers[_outputIndex] = &_output_gpu;
  cudaMalloc(&_cuda_buffers[_inputIndex], _sizeof_in);
  cudaMalloc(&_cuda_buffers[_outputIndex], _sizeof_out);

  // Use CUDA streams to manage the concurrency of copying and executing
  cudaStreamCreate(&_cuda_stream);

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
retCode NetTRT::infer(const cv::Mat& image, cv::Mat& mask, const bool verbose) {
  // Check if image has something
  if (!image.data) {
    std::cout << "Could find content in the image" << std::endl;
    return CNN_FAIL;
  }

  // start total counter
  auto start_time_total = std::chrono::high_resolution_clock::now();

  // Get dimensions and check that it has proper channels
  static unsigned int num_classes = _cfg_data["label_remap"].size();
  static unsigned int d = _cfg_data["img_prop"]["depth"].as<unsigned int>();
  static unsigned int w = _cfg_data["img_prop"]["width"].as<unsigned int>();
  static unsigned int h = _cfg_data["img_prop"]["height"].as<unsigned int>();
  unsigned int cv_img_d = image.channels();
  assert(cv_img_d == d);

  // Set up inputs to run the graph
  // First convert to proper size, format, and normalize
  cv::Mat norm_image;
  cv::resize(image, norm_image, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  norm_image.convertTo(norm_image, CV_32FC3);
  norm_image = (norm_image - 128.0f) / 128.0f;

  // WATCH OUT! CUDA takes channel first, opencv is channel last
  // split in B, G, R
  std::vector<cv::Mat> norm_image_chw(d);
  cv::split(norm_image, norm_image_chw);
  // copy the B, G and, R data into a contiguous array
  std::vector<float> norm_image_chw_data;
  for (unsigned int ch = 0; ch < d; ++ch) {
    if (norm_image_chw[ch].isContinuous()) {
      norm_image_chw_data.insert(norm_image_chw_data.end(),
                                 (float*)norm_image_chw[ch].datastart,
                                 (float*)norm_image_chw[ch].dataend);
    } else {
      for (unsigned int y = 0; y < h; ++y) {
        norm_image_chw_data.insert(norm_image_chw_data.end(),
                                   norm_image_chw[ch].ptr<float>(y),
                                   norm_image_chw[ch].ptr<float>(y) + w);
      }
    }
  }

  // Run the graph
  // start inference counter
  auto start_time_inference = std::chrono::high_resolution_clock::now();

  // Copy Input Data to the GPU memory
  cudaMemcpyAsync(_cuda_buffers[_inputIndex], norm_image_chw_data.data(),
                  _sizeof_in, cudaMemcpyHostToDevice, _cuda_stream);

  // Enqueue the op
  _context->enqueue(1, _cuda_buffers, _cuda_stream, nullptr);

  // Copy Output Data to the CPU memory
  std::vector<float> output_chw(_size_in_pix * num_classes);
  cudaMemcpyAsync(output_chw.data(), _cuda_buffers[_outputIndex], _sizeof_out,
                  cudaMemcpyDeviceToHost, _cuda_stream);

  // sync point
  cudaStreamSynchronize(_cuda_stream);

  // elapsed_inference time
  auto elapsed_inference =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start_time_inference)
          .count();

  // Process the output with map
  // WATCH OUT! CUDA gives channel first, opencv is channel last
  // Convert to vector of unidimensional mats and merge

  std::vector<cv::Mat> output_cMats(num_classes);
  cv::Mat output;  // merged output (easier and faster argmax)
  for (unsigned int c = 0; c < num_classes; ++c) {
    float* slice_p = &output_chw[c * _size_in_pix];
    output_cMats[c] = cv::Mat(cv::Size(w, h), CV_32FC1, slice_p);
  }
  cv::merge(output_cMats, output);

  // for each pixel, calculate the argmax
  cv::Mat argmax(cv::Size(w, h), CV_32SC1);
  for (unsigned int y = 0; y < h; ++y) {
    int* row_argmax = argmax.ptr<int>(y);
    float* row_c = output.ptr<float>(y);
    for (unsigned int x = 0; x < w; ++x) {
      float max = row_c[x * num_classes];
      int max_c = 0;
      for (unsigned int ch = 1; ch < num_classes; ++ch) {
        if (row_c[x * num_classes + ch] > max) {
          max = row_c[x * num_classes + ch];
          max_c = ch;
        }
      }
      row_argmax[x] = max_c;
    }
  }

  // elapsed_total time
  auto elapsed_total =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start_time_total)
          .count();

  if (verbose) {
    std::cout << "Successfully run prediction from engine." << std::endl;
    std::cout << "Time to infer: " << elapsed_inference << "ms." << std::endl;
    std::cout << "Time in total: " << elapsed_total << "ms." << std::endl;
  }

  // convert to 8U and put in mask
  argmax.convertTo(mask, CV_8U);

  return CNN_OK;
}

/**
 * @brief      Set verbosity level for backend execution
 *
 * @param[in]  verbose  True is max verbosity, False is no verbosity.
 *
 * @return     Exit code.
 */
retCode NetTRT::verbosity(const bool verbose) {
  _logger.set_verbosity(verbose);
  return CNN_OK;
}

} /* namespace bonnet */
