// STD
#include <unistd.h>
#include <string>

// net stuff
#include "bonnet_handler.hpp"
#ifdef TF_AVAIL
#include <netTF.hpp>
#endif  // TF_AVAIL
#ifdef TRT_AVAIL
#include <netTRT.hpp>
#endif  // TRT_AVAIL
#if (!defined TF_AVAIL) && (!defined TRT_AVAIL)
#error("At least TF OR TensorRT must be installed")
#endif

namespace bonnet {

/*!
 * Constructor.
 *
 * @param      nodeHandle  the ROS node handle.
 */
netHandler::netHandler(ros::NodeHandle& nodeHandle)
    : node_handle_(nodeHandle), it_(nodeHandle) {
  // Try to read the necessary parameters
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }

  // Subscribe to images to infer
  img_subscriber_ =
      it_.subscribe(img_subscriber_topic_, 1, &netHandler::imageCallback, this);

  // Advertise our topics
  bgr_publisher_ = it_.advertise(bgr_publisher_topic_, 1);
  mask_publisher_ = it_.advertise(mask_publisher_topic_, 1);
  color_mask_publisher_ = it_.advertise(color_mask_publisher_topic_, 1);

  ROS_INFO("Successfully launched node.");
}

/**
  * @brief      Initialize the Handler
  *
  * @return     Error code
  */
retCode netHandler::init() {
  // before doing anything, make sure we have a slash at the end of path
  model_path_ += "/";

  // Try to get the cofig nodes
  try {
    cfg_train_ = YAML::LoadFile(model_path_ + "train.yaml");
    cfg_net_ = YAML::LoadFile(model_path_ + "net.yaml");
    cfg_data_ = YAML::LoadFile(model_path_ + "data.yaml");
    cfg_nodes_ = YAML::LoadFile(model_path_ + "nodes.yaml");
  } catch (YAML::Exception& ex) {
    ROS_ERROR("Can't open one of the config files from the model path");
    ROS_ERROR("%s", ex.what());
    return CNN_FAIL;
  }

  // Init net according to selected backend
  if (backend_ == "tf") {
#ifdef TF_AVAIL
    // Get the full path to the model
    pb_model_path_ = model_path_ + model_ + ".pb";
    ROS_INFO("Model path: %s", pb_model_path_.c_str());
    if (access(pb_model_path_.c_str(), R_OK) != 0) {
      ROS_ERROR("Model pb file has no permission or does not exist!");
      return CNN_FATAL;
    }

    // define net
    net_ =
        new NetTF(pb_model_path_, cfg_train_, cfg_net_, cfg_data_, cfg_nodes_);

    // define verbosity
    net_->verbosity(verbose_);
#else
    ROS_ERROR("Tensorflow is supported, but build couldn't find it");
    return CNN_FATAL;
#endif  // TF_AVAIL

  } else if (backend_ == "trt") {
#ifdef TRT_AVAIL
    // Get the full path to the model
    pb_model_path_ = model_path_ + "optimized_tRT.uff";
    ROS_INFO("Model path: %s", pb_model_path_.c_str());
    if (access(pb_model_path_.c_str(), R_OK) != 0) {
      ROS_ERROR("Model pb file has no permission or does not exist!");
      return CNN_FATAL;
    }

    // define net
    net_ =
        new NetTRT(pb_model_path_, cfg_train_, cfg_net_, cfg_data_, cfg_nodes_);

    // define verbosity
    net_->verbosity(verbose_);
#else
    ROS_ERROR("TensorRT is supported, but build couldn't find it");
    return CNN_FATAL;
#endif  // TRT_AVAIL
  } else {
    ROS_ERROR(
        "Tensorflow (tf) and TensorRT (trt) backends are the only one "
        "implemented right now");
    return CNN_FATAL;
  }

  // Initialize net in proper device
  retCode status = net_->init(device_);
  if (status != CNN_OK) {
    ROS_ERROR("Failed to intialize CNN.");
    return status;
  }

  return CNN_OK;
}

/*!
 * Destructor.
 */
netHandler::~netHandler() {
  if (net_) {
    delete net_;
  }
}

bool netHandler::readParameters() {
  if (!node_handle_.getParam("image_topic", img_subscriber_topic_) ||
      !node_handle_.getParam("bgr_topic", bgr_publisher_topic_) ||
      !node_handle_.getParam("mask_topic", mask_publisher_topic_) ||
      !node_handle_.getParam("color_mask_topic", color_mask_publisher_topic_) ||
      !node_handle_.getParam("model_path", model_path_) ||
      !node_handle_.getParam("model", model_) ||
      !node_handle_.getParam("verbose", verbose_) ||
      !node_handle_.getParam("device", device_) ||
      !node_handle_.getParam("backend", backend_))
    return false;
  return true;
}

void netHandler::imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {
  if (verbose_) {
    // report that we got something
    ROS_INFO("Image received.");
    ROS_INFO("Image encoding: %s", img_msg->encoding.c_str());
  }

  // Get the image
  cv_bridge::CvImageConstPtr cv_img;
  cv_img = cv_bridge::toCvShare(img_msg);

  // change to bgr according to encoding
  cv::Mat cv_img_bgr(cv_img->image.rows, cv_img->image.cols, CV_8UC3);
  ;
  if (img_msg->encoding == "bayer_rggb8") {
    if (verbose_) ROS_INFO("Converting BAYER_RGGB8 to BGR for CNN");
    cv::cvtColor(cv_img->image, cv_img_bgr, cv::COLOR_BayerBG2BGR);
  } else if (img_msg->encoding == "bgr8") {
    if (verbose_) ROS_INFO("Converting BGR8 to BGR for CNN");
    cv_img_bgr = cv_img->image;
  } else if (img_msg->encoding == "rgb8") {
    if (verbose_) ROS_INFO("Converting RGB8 to BGR for CNN");
    cv::cvtColor(cv_img->image, cv_img_bgr, cv::COLOR_RGB2BGR);
  } else {
    if (verbose_) ROS_ERROR("Colorspace conversion non implemented. Skip...");
    return;
  }

  // Infer with net
  cv::Mat mask_argmax;
  net_->infer(cv_img_bgr, mask_argmax, verbose_);

  // Resize to the input original size in order to get pixel correspondence
  cv::resize(mask_argmax, mask_argmax,
             cv::Size(cv_img_bgr.cols, cv_img_bgr.rows), 0, 0,
             cv::INTER_NEAREST);

  // Send the mask
  sensor_msgs::ImagePtr mask_msg =
      cv_bridge::CvImage(img_msg->header, "mono8", mask_argmax).toImageMsg();
  mask_publisher_.publish(mask_msg);

  // Send the color mask
  cv::Mat mask_bgr(mask_argmax.rows, mask_argmax.cols, CV_8UC3);
  net_->color(mask_argmax, mask_bgr, verbose_);
  sensor_msgs::ImagePtr color_mask_msg =
      cv_bridge::CvImage(img_msg->header, "bgr8", mask_bgr).toImageMsg();
  color_mask_publisher_.publish(color_mask_msg);

  // Echo the image as bgr
  sensor_msgs::ImagePtr bgr_msg =
      cv_bridge::CvImage(img_msg->header, "bgr8", cv_img_bgr).toImageMsg();
  bgr_publisher_.publish(bgr_msg);
}

}  // namespace chatter
