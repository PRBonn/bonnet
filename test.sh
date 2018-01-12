#!/bin/bash

# env vars
export PATH=/usr/local/cuda-9.0/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# dependencies
apt-get install -y python-pip python-empy
pip install -U pip catkin-tools trollius
source /opt/ros/kinetic/setup.bash

# build
cd deploy_cpp/standalone # build standalone apps
mkdir -p build
cd build
cmake ..
make -j
cd ../../../
mkdir -p ~/catkin_ws/src/bonnet/ # build ros nodes
cp -r ./* ~/catkin_ws/src/bonnet/
cd ~/catkin_ws
catkin init
catkin build
