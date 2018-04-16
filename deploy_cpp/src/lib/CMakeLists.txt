cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(bonnet_core)

# handle debug vs release mode
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")

## Use C++11
add_definitions(-std=c++11 -Wfatal-errors)

# check flags
message("Build type: ${CMAKE_BUILD_TYPE}")

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    bonnet_core
  CATKIN_DEPENDS
#  DEPENDS
)

# set flags for Tensorflow and TensorRT availability
option(TF_AVAIL "Tensorflow available" OFF)
option(CUDA_AVAIL "CUDA available" OFF)
option(TRT_AVAIL "TensorRT available" OFF)

## Find libraries
# try the shared tensorflow
find_package(TensorflowCC COMPONENTS Shared)
if(TensorflowCC_Shared_FOUND)
  get_property(TF_LIBS TARGET TensorflowCC::Shared PROPERTY INTERFACE_LINK_LIBRARIES)
  get_property(TF_INCLUDE_DIRS TARGET TensorflowCC::Shared PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  message("Tensorflow Libs: ${TF_LIBS}")
  message("Tensorflow Headers: ${TF_INCLUDE_DIRS}")
  add_definitions(-DTF_AVAIL)
  set(TF_AVAIL ON)
else()
  message("Tensorflow_cc shared library NOT FOUND")
  set(TF_AVAIL OFF)
endif()
message("TF_AVAIL ${TF_AVAIL}\n")

# try to find CUDA
find_package(CUDA)
if (CUDA_FOUND)
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)
message("CUDA_AVAIL ${CUDA_AVAIL}")

# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
if(NVINFER AND NVPARSERS)
  message("TensorRT available!")
  message("NVINFER: ${NVINFER}")
  message("NVPARSERS: ${NVPARSERS}")
  set(TRT_AVAIL ON)
  add_definitions(-DTRT_AVAIL)
else()
  message("TensorRT NOT Available")
  set(TRT_AVAIL OFF)
endif()
message("TRT_AVAIL ${TRT_AVAIL}\n")

# get opencv
find_package(OpenCV)
if(OpenCV_FOUND)
  message("OpenCV Libs: ${OpenCV_LIBS}")
  message("OpenCV Headers: ${OpenCV_INCLUDE_DIRS}\n")
else()
  message(FATAL_ERROR "OpenCV NOT FOUND\n")
endif()

# libyaml-cpp
find_package (yaml-cpp)
if (yaml-cpp_FOUND)
  message("YAML Libs: ${YAML_CPP_LIBRARIES}")
  message("YAML Headers: ${YAML_CPP_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "yaml-cpp NOT FOUND\n")
endif (yaml-cpp_FOUND)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

## Declare a cpp library
# define the network library (compiling tf and trt conditionally)
set(bonnet_core_SOURCES src/net.cpp src/bonnet.cpp)
set(bonnet_core_INCLUDES include
                         ${OpenCV_INCLUDE_DIRS}
                         ${YAML_CPP_INCLUDE_DIR})
set(bonnet_core_LIBRARIES ${OpenCV_LIBS}
                          ${YAML_CPP_LIBRARIES})
if(TF_AVAIL)
  list(APPEND bonnet_core_SOURCES src/netTF.cpp)
  list(APPEND bonnet_core_INCLUDES ${TF_INCLUDE_DIRS})
  list(APPEND bonnet_core_LIBRARIES ${TF_LIBS})
endif()
if(TRT_AVAIL AND CUDA_AVAIL)
  list(APPEND bonnet_core_SOURCES src/netTRT.cpp)
  list(APPEND bonnet_core_INCLUDES ${CUDA_INCLUDE_DIRS})
  list(APPEND bonnet_core_LIBRARIES ${CUDA_LIBRARIES}
                                    nvinfer
                                    nvparsers)
endif()
add_library(bonnet_core ${bonnet_core_SOURCES})
target_include_directories(bonnet_core PUBLIC ${bonnet_core_INCLUDES})
target_link_libraries(bonnet_core PUBLIC ${bonnet_core_LIBRARIES})

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS bonnet_core
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)