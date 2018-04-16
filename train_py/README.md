# Bonnet (py): Tensorflow Convolutional Semantic Segmentation pipeline

## Description

- Contains Python code to easily build ConvNet Graphs in TensorFlow,
train, and generate the trained models used for deployment. This way the
interface with TensorFlow can use the more complete Python API and we can easily
work with files to augment datasets and so on.

## Dependencies

- Python stuff (and tf profiler):
  
```sh
  $ sudo apt install libav-tools ffmpeg libcupti-dev
  $ sudo pip3 install -r requirements.txt
  $ export PYTHONPATH=/usr/local/lib/python3.5/dist-packages/cv2/:$PYTHONPATH # Needed if you have ROS Kinetic installed
```

- Tensorflow (Follow link, it is complete for install of CUDA, CUDNN, etc): [Link](https://www.tensorflow.org/install/install_linux)

- If the tensorRT backend to run with higher performance is desired you neet to install it:
  - TensorRT [Download](https://developer.nvidia.com/nvidia-tensorrt-download)
  - PyCUDA [Link](https://wiki.tiker.net/PyCuda/Installation/Linux)

## Usage

### Train a network

```sh
$ ./cnn_train.py -d data.yaml -n net.yaml -t train.yaml -l /tmp/path/to/log/ -p /tmp/path/to/pretrained
```

- _"cnn_train.py"_ contains the code necessary to load the config files and interact with the dataset and network classes, in order to train the system.
- _"data.yaml", "net.yaml", and "train.yaml"_ are the configuration files needed to pass all parameters to the network and dataset. Examples for them are in the _"cfg/"_ folder.
- You can specify the log directory, where the trained model, the tensorboard log, and some predictions will be saved during training.
- You can specify the path to an old log directory, containing a pretrained model, only if it was created with this pipeline.

### Use a network

###### To get visual predictions:

```sh
$ ./cnn_use.py -l /tmp/path/to/log/ -p /tmp/path/to/pretrained -i /path/to/image
```

  - _"cnn_use.py"_ uses the pretrained model from disk, it builds the whole computational graph, and then it calculates the mask for each image. Finally, it saves all predictions to the log path.
  - There is no need to specify cfg files, because they are contained in the pretrained model folder, but they can be specified to change parameters, as long as the graph is still valid.
  - The image path can be a single image, or a list of images. For example, for all images in a folder, do _"-i /path/\*"_
  - The _"--verbose"_ option calculates a trace file for each run, which assesses the runtime in GPU, and also plots the results on screen.

```sh
$ ./cnn_video.py -v /tmp/path/to/video.mp4 -p /tmp/path/to/pretrained 
```

  - _"cnn_video.py"_ uses the pretrained model from disk, it builds the whole computational graph, and then it calculates the mask for each frame in the video.
  - There is no need to specify cfg files, because they are contained in the pretrained model folder, but they can be specified to change parameters, as long as the graph is still valid.


###### To get probability predictions:

```sh
$ ./cnn_use.py -l /tmp/path/to/log/ -p /tmp/path/to/pretrained -i /path/to/image --probs
```

  - Same as before but now outputs all probability maps to ".mat" files in the log folder.

###### To get central feature map:

```sh
$ ./cnn_use.py -l /tmp/path/to/log/ -p /tmp/path/to/pretrained -i /path/to/image --code
```

  - Same as before but now outputs the encoded feature map to ".txt" files in the log folder, reshaped as single row vectors with 8 decimal values.

###### To freeze a trained graph (for feeding into the cpp deployment pipeline):

```sh
$ ./cnn_freeze.py -p /tmp/path/to/pretrained -l /tmp/path/to/log
```

  - Freezes the graph into a protobuf and it gives you 4 models:
    - frozen_nchw.pb: network in nchw format, which is better for GPU exec.
    - frozen_nhwc.pb: network in nhwc format, which is sometimes better for CPU exec.
    - optimized.pb:   network optimized for inference with the tensorflow library.
    - optimized_tRT.pb: network optimized for inference with tensorRT3 (NVIDIA)
    - optimized_tRT.uff: network optimized for inference with tensorRT3 (NVIDIA) 
    - quantized.pb:   network quantized to 8-bit, which is good for mobile (uncalibrated).
    - Lot's of config yaml files that you may need in case you want to debug something.
    - The graph as a tensorboard log.

###### To use the network from Tensorflow with TensorRT.

We provide _*cnn_use_pb_tensorRT.py*_ and _*cnn_video_pb_tensorRT.py*_, which
are analogous to their frozen pb tensorflow counterparts, so you can use them in
the same way but with the other backend :) If this works, then the cpp node
should work as well.