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

// standard
#include <iostream>

// abstract net
#include <net.hpp>

// Nvidia stuff
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvUffParser.h"

namespace nvinf = nvinfer1;
namespace uffpar = nvuffparser;

namespace bonnet {

/**
 * @brief      Class for TensorRT network inference.
 */
class NetTRT : public Net {
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
  NetTRT(const std::string& model_path, const YAML::Node& cfg_train,
         const YAML::Node& cfg_net, const YAML::Node& cfg_data,
         const YAML::Node& cfg_nodes);

  /**
   * @brief      Destroys the object.
   */
  virtual ~NetTRT();

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

 protected:
  /**
   * @brief      Class for logger. Will be used to set the verbosity.
   */
  class Logger : public nvinf::ILogger {
   public:
    void set_verbosity(bool verbose) { _verbose = verbose; }
    void log(Severity severity, const char* msg) override {
      if (_verbose) std::cout << msg << std::endl;
    }

   private:
    bool _verbose = false;
  };

 private:
  // Cuda and TensorRT stuff
  Logger _logger;                       // logger for TensorRT engine
  nvinf::IBuilder* _builder;            // builder for TensorRT engine
  bool _fp16;                           // platform has fp16 support?
  nvinf::DataType _net_data_type;       // depends on _fp16
  nvinf::INetworkDefinition* _network;  // where to populate network
  uffpar::IUffParser* _parser;  // parser for uff model (from frozen pb in py)
  nvinf::ICudaEngine* _engine;  // cuda engine to run the model
  nvinf::IExecutionContext* _context;         // context to launch the kernels
  int _inputIndex, _outputIndex;              // bindings for cuda i/o
  nvinf::Dims _inputDims, _outputDims;        // dimensions of input and output
  int _size_in_pix, _sizeof_in, _sizeof_out;  // size for cuda malloc
  cudaStream_t _cuda_stream;  // cuda streams handles copying to/from GPU

  // pointers to GPU memory of input and output
  float* _input_gpu;
  int* _output_gpu;
  void* _cuda_buffers[2];

  // graph nodes for i/o
  std::string _input_node;   // name of input node in tf graph
  std::string _output_node;  // name of output node in tf graph
};

} /* namespace bonnet */
