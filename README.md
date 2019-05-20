# Bonnet: An Open-Source Training and Deployment Framework for Semantic Segmentation in Robotics.

[![Build Status](https://travis-ci.org/PRBonn/bonnet.svg?branch=master)](https://travis-ci.org/PRBonn/bonnet)

By [Andres Milioto](http://www.ipb.uni-bonn.de/people/andres-milioto/) @ University of Bonn.

**__(for the new Pytorch version, go [here](https://github.com/PRBonn/bonnetal))__**

![Image of cityscapes](https://image.ibb.co/i5tEQR/CITY.png)
Cityscapes Urban Scene understanding.
- [Standalone Video Predictor - 512x256px](https://youtu.be/RXK_eYO_i08)
- [Standalone Video Predictor - 768x384px (inception model)](https://youtu.be/3qD-GR87usw)
- [Standalone Video Predictor - 1024x512px](https://youtu.be/tfeFHCq6YJs)

![Image of Persons](https://image.ibb.co/nJRa3c/github_bruno.png)
Person Segmentation
- [Uptown funk (inception model)](https://photos.app.goo.gl/Ls3MkCo2Cwhc6pVT2)
- [Can't stop the feeling (inception model)](https://photos.app.goo.gl/Mi4N4zLrTyyDRqv48)


![Image of cwc](https://image.ibb.co/fcKXC6/CWC.png)
Crop vs. Weed Semantic Segmentation.
- [ROS node prediction - Video](https://youtu.be/-XgxiC04hUI)


## Description

This code provides a framework to easily add architectures and datasets, in order to 
train and deploy CNNs for a robot. It contains a full training pipeline in python
using Tensorflow and OpenCV, and it also some C++ apps to deploy a frozen
protobuf in ROS and standalone. The C++ library is made in a way which allows to
add other backends (such as TensorRT and MvNCS), but only Tensorflow and TensorRT
are implemented for now. For now, we will keep it this way because we are mostly
interested in deployment for the Jetson and Drive platforms, but if you have a specific
need, we accept pull requests!

The networks included is based of of many other architectures
(see below), but not exactly a copy of any of them. As seen in the videos, they run very fast in
both GPU and CPU, and they are designed with performance in mind, at the cost of
a slight accuracy loss. Feel free to use it as a model to implement your own
architecture.

All scripts have been tested on the following configurations:
- x86 Ubuntu 16.04 with an NVIDIA GeForce 940MX GPU (nvidia-384, CUDA9, CUDNN7, TF 1.7, TensorRT3)
- x86 Ubuntu 16.04 with an NVIDIA GTX1080Ti GPU (nvidia-375, CUDA9, CUDNN7, TF 1.7, TensorRT3)
- x86 Ubuntu 16.04 and 14.04 with no GPU (TF 1.7, running on CPU in NHWC mode, no TensorRT support)
- Jetson TX2 (full Jetpack 3.2)

We also provide a Dockerfile to make it easy to run without worrying about the dependencies, which is based on the official nvidia/cuda image containing cuda9 and cudnn7. In order to build and run this image with support for X11 (to display the results), you can run this in the repo root directory ([nvidia-docker](https://github.com/NVIDIA/nvidia-docker) should be used instead of vainilla docker):

```sh
  $ docker pull tano297/bonnet:cuda9-cudnn7-tf17-trt304
  $ nvidia-docker build -t bonnet .
  $ nvidia-docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v /home/$USER/data:/shared --net=host --pid=host --ipc=host bonnet /bin/bash
```

_-v /home/$USER/data:/share_ can be replaced to point to wherever you store the data and trained models, in order to include the data inside the container for inference/training.

#### Deployment

- _/deploy_cpp_ contains C++ code for deployment on robot of the full pipeline,
which takes an image as input and produces the pixel-wise predictions
as output, and the color masks (which depend on the problem). It includes both
standalone operation which is meant as an example of usage and build, and a ROS
node which takes a topic with an image and outputs 2 topics with the labeled mask
and the colored labeled mask.

- Readme [here](deploy_cpp/README.md)

#### Training

- _/train_py_ contains Python code to easily build CNN Graphs in Tensorflow,
train, and generate the trained models used for deployment. This way the
interface with Tensorflow can use the more complete Python API and we can easily
work with files to augment datasets and so on. It also contains some apps for using
models, which includes the ability to save and use a frozen protobuf, and to use
the network using TensorRT, which reduces the time for inference when using NVIDIA
GPUs.

- Readme [here](train_py/README.md)

#### Pre-trained models

These are some models trained on some sample datasets that you can use with the trainer and deployer,
but if you want to take time to write the parsers for another dataset (yaml file with classes and colors + python script to
put the data into the standard dataset format) feel free to create a pull request.

If you don't have GPUs and the task is interesting for robots to exploit, I will
gladly train it whenever I have some free GPU time in our servers.

- Cityscapes:
  - 512x256 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/city_512.tar.gz)
  - 768x384 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/city_768_inception.tar.gz) (inception-like model)
  - 768x384 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/city_768_mobilenets.tar.gz) (mobilenets-like model)
  - 1024x512 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/city_1024.tar.gz)
- Synthia:
  - 512x384 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/synthia_512.tar.gz)
  - 960x720 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/synthia_960.tar.gz)

- Persons (+coco people):
  - 512x512 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/persons_512.tar.gz)

- Crop-Weed (CWC):
  - 512x384 [Link](http://www.ipb.uni-bonn.de/html/projects/bonnet/pretrained-models/v0.2/cwc_512.tar.gz)

## License

#### This software

Bonnet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Bonnet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

#### Pretrained models

The pretrained models with a specific dataset keep the copyright of such dataset.

- Cityscapes: [Link](https://www.cityscapes-dataset.com)
- Synthia: [Link](http://synthia-dataset.net)
- Persons: [Link](https://supervise.ly)
- Coco: [Link](http://cocodataset.org/#home)
- Crop-Weed (CWC): [Link](http://www.ipb.uni-bonn.de/data/sugarbeets2016/)

## Citation

If you use our framework for any academic work, please cite its [paper](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019icra.pdf).

```
@InProceedings{milioto2019icra,
author = {A. Milioto and C. Stachniss},
title = {{Bonnet: An Open-Source Training and Deployment Framework for Semantic Segmentation in Robotics using CNNs}},
booktitle = {Proc. of the IEEE Intl. Conf. on Robotics \& Automation (ICRA)},
year = 2019,
codeurl = {https://github.com/Photogrammetry-Robotics-Bonn/bonnet},
videourl = {https://www.youtube.com/watch?v=tfeFHCq6YJs},
}
```

Our networks are strongly based on the following architectures, so if you
use them for any academic work, please give a look at their papers and cite them
if you think proper:

- SegNet: [Link](https://arxiv.org/abs/1511.00561)
- E-Net: [Link](https://arxiv.org/abs/1606.02147)
- ERFNet: [Link](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)
- PSPNet [Link](https://arxiv.org/abs/1612.01105)

## Other useful GitHub's:
- [OpenAI Checkpointed Gradients](https://github.com/openai/gradient-checkpointing). Useful
implementation of checkpointed gradients to be able to fit big models in GPU memory without sacrificing
runtime.
- [Queueing tool](https://github.com/alexanderrichard/queueing-tool): Very nice
queueing tool to share GPU, CPU and Memory resources in a multi-GPU environment.
- [Tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc): Very useful repo
to compile Tensorflow either as a shared or static library using CMake, in order
to be able to compile our C++ apps against it.

## Contributors

Milioto, Andres
- [University of Bonn](http://www.ipb.uni-bonn.de/people/andres-milioto/)
- [Linkedin](https://www.linkedin.com/in/amilioto/)
- [ResearchGate](https://www.researchgate.net/profile/Andres_Milioto)
- [Google Scholar](https://scholar.google.de/citations?user=LzsKE7IAAAAJ&hl=en)

Special thanks to [Philipp Lottes](http://www.ipb.uni-bonn.de/people/philipp-lottes/)
for all the work shared during the last year, and to [Olga Vysotka](http://www.ipb.uni-bonn.de/people/olga-vysotska/) and
[Susanne Wenzel](http://www.ipb.uni-bonn.de/people/susanne-wenzel/) for beta testing the 
framework :)

## Acknowledgements

This work has partly been supported by the German Research Foundation under Germany's Excellence Strategy, EXC-2070 - 390732324 (PhenoRob). We also thank NVIDIA Corporation for providing a Quadro P6000 GPU partially used to develop this framework.

## TODOs

- Merge [Crop-weed CNN with background knowledge](https://arxiv.org/pdf/1709.06764.pdf) into this repo.
- Make multi-camera ROS node that exploits batching to make inference faster than sequentially.
- Movidius Neural Stick C++ backends (plus others as they become available).
- Inference node to show the classes selectively (e.g. with some qt visual GUI)
