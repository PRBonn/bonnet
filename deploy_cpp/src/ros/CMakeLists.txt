cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(bonnet_run)

# handle debug vs release mode
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Use C++11
add_definitions(-std=c++11 -Wfatal-errors)

# check flags
message("Build type: ${CMAKE_BUILD_TYPE}")

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    sensor_msgs
    std_msgs
    cv_bridge
    image_transport
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
    sensor_msgs
    std_msgs
#  DEPENDS
)

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
  message("OpenCV Libs: ${OpenCV_LIBRARIES}")
  message("OpenCV Headers: ${OpenCV_INCLUDE_DIRS}\n")
else()
  message(FATAL_ERROR "OpenCV NOT FOUND\n")
endif()

# libyaml-cpp
find_package (yaml-cpp)
if (yaml-cpp_FOUND)
  message("YAML Libs: ${YAML_CPP_LIBRARIES}")
  message("YAML Headers: ${YAML_CPP_INCLUDE_DIR}\n")
else()
  message(FATAL_ERROR "yaml-cpp NOT FOUND\n")
endif (yaml-cpp_FOUND)

find_package(bonnet_core)
if(bonnet_core_FOUND)
  message("bonnet_core Libs: ${bonnet_core_LIBRARIES}")
  message("bonnet_core Headers: ${bonnet_core_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "bonnet_core Not Found!\n")
endif()
                      

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
list(APPEND bonnet_node_INCLUDES include)
if(TF_AVAIL)
  list(APPEND bonnet_node_INCLUDES ${TF_INCLUDE_DIRS})
  list(APPEND bonnet_node_LIBRARIES ${TF_LIBS})
endif()
if(TRT_AVAIL AND CUDA_AVAIL)
  list(APPEND bonnet_node_INCLUDES ${CUDA_INCLUDE_DIRS})
  list(APPEND bonnet_node_LIBRARIES ${CUDA_LIBRARIES}
                                    nvinfer
                                    nvparsers)
endif()

include_directories(
  ${bonnet_node_INCLUDES}
  ${bonnet_core_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

## Declare cpp executables
add_executable(bonnet_node
  src/bonnet_node.cpp
  src/bonnet_handler.cpp
)

target_link_libraries(bonnet_node
  ${bonnet_node_LIBRARIES}
  ${bonnet_core_LIBRARIES}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS bonnet_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  ## Add gtest based cpp test target and link libraries!
endif()
