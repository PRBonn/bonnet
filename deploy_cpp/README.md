# Bonnet (cpp): Tensorflow Convolutional Semantic Segmentation pipeline

## Description

- Contains C++ code for deployment on robot of the full pipeline,
which takes a camera device as input and produces the pixel-wise predictions
as output (which depend on the problem).

## Dependencies

#### To use with tensorflow backend
- Install Bazel [link](https://docs.bazel.build/versions/master/install-ubuntu.html#install-on-ubuntu)

- Tensorflow Cpp using shared library cmake install [External Install](https://github.com/FloopCZ/tensorflow_cc)

#### To use with TensorRT

- If the tensorRT backend to run with higher performance is desired you neet to install it:
  - TensorRT [Download](https://developer.nvidia.com/nvidia-tensorrt-download)

**_IMPORTANT_**: Tensorflow and TensorRT don't need each other in C++ mode. This is important
when deploying in an embedded device, such as the Jetson, as Tensorflow is slow and cumbersome
to build, and it is not necessary, considering that the tensorRT backend works much faster. 
Therefore: **THE CMAKE SCRIPTS FLAG THE NODES WHICH BACKENDS ARE AVAILABLE, AND IF ONE
IS NOT INSTALLED, IT IS NOT USED.**

#### Extra stuff

- We use catkin tools for building the library and the apps (both ROS and standalone):

```sh
  $ sudo pip install catkin_tools trollius
```

- Boost and yaml (if you have ROS Kinetic installed, this is not necessary): 

```sh
  $ sudo apt install libboost-all-dev libyaml-cpp-dev
```

- Opencv3: [Link](http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html) (if you have ROS Kinetic installed, this is not necessary)

- If you want to use the ROS node, of course, you need to install [ROS](http://wiki.ros.org/ROS/Installation) We tested on Kinetic, and CI docker is on kinetic as well :)


## Usage

#### Freezing your model to deploy using C++ backends

Refer to the python train [readme](../train_py/README.md) section to see model freezing before starting to work!

```sh
$ cd train_py/
$ ./cnn_freeze.py -p /tmp/path/to/pretrained -l /tmp/path/to/log
```

If you only want to deploy a model without the dependencies of the python section, you can try the docker container we provide in the main [readme](../README.md).

#### Standalone examples

We use catkin tools. These are example files to check the usage of your frozen models and your tensorflow install.

###### Build

```sh
$ cd bonnet/deploy_cpp
$ catkin init
$ catkin build bonnet_standalone
```

###### Use

The _"cnn_use_pb"_ app takes a frozen protobuf and images and predicts the output masks. 

```sh
$ ./build/bonnet_standalone/cnn_use_pb -p /tmp/path/to/pretrained -i /path/to/image -l /tmp/path/to/log/ -b trt/tf
```

  - _"cnn_use_pb"_ uses the frozen tensorflow model from disk and calculates the mask for each image. Finally, it saves all predictions to the log path.
  - The image path can be a single image, or a list of images. For example, for all images in a folder, do _"-i /path/\*"_
  - The "_-b_" flag specifies the backend (tf is tensorflow, trt is TensorRT)
  - The _"--verbose"_ option outputs all tensorflow debug commands, calculates inference time, and also plots the results on screen.
  - For more information run with the _"-h"_ argument

The _"cnn_video_pb"_ app takes a frozen protobuf and a video and predicts the output masks. 

```sh
$ ./build/bonnet_standalone/cnn_video_pb -p /tmp/path/to/pretrained -v /path/to/video -l /tmp/path/to/log/ -b trt/tf
```

  - _"cnn_video_pb"_ uses the frozen tensorflow model from disk and calculates the mask for each frame. Finally, it saves all predictions to the log path.
  - The image path can be a single image, or a list of images. For example, for all images in a folder, do _"-i /path/\*"_
  - The "_-b_" flag specifies the backend (tf is tensorflow, trt is TensorRT)
  - The _"--verbose"_ option outputs all tensorflow debug commands, calculates inference time, profile, and also plots the results on screen.
  - For more information run with the _"-h"_ argument

The _"session"_ app starts a tensorflow session and outputs if it was successful (as a test)

```sh
$ ./build/bonnet_standalone/session
```

  - _"session"_ is mainly for checking if the tensorflow install went well and if you can see the GPU, as it will output its success and the visible devices.
If a GPU was found you should see something like this:

```sh
$ ./build/bonnet_standalone/session

2017-12-18 13:12:40.776389: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce 940MX major: 5 minor: 0 memoryClockRate(GHz): 1.2415
pciBusID: 0000:02:00.0
totalMemory: 1.96GiB freeMemory: 782.06MiB
2017-12-18 13:12:40.776407: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce 940MX, pci bus id: 0000:02:00.0, compute capability: 5.0)
Session successfully created.

```

#### ROS

The node reads from the image topic configured in the config file and outputs both the cross entropy
mask (0 to num_classes-1) and the color mask to the topics configured in the config file as well.

###### Build

To build, just put this project in your catkin workspace (or make a symlink to it) and build using catkin tools:

```
$ ln -s /path/to/bonnet/git/repo ~/catkin_ws/src/bonnet
$ cd ~/catkin_ws 
$ catkin build
$ source devel/setup.bash
```

###### Launch

```sh
$ roslaunch bonnet_run bonnet_run.launch
```