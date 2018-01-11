#pragma once

// ROS
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

// Network
#include <net.hpp>

namespace bonnet {

/*!
 * Main class for the node to handle the ROS interfacing.
 */
class netHandler {
 public:
  /*!
   * Constructor.
   *
   * @param      nodeHandle  the ROS node handle.
   */
  netHandler(ros::NodeHandle& nodeHandle);

  /*!
   * Destructor.
   */
  virtual ~netHandler();

  /**
   * @brief      Initialize the Handler
   *
   * @return     Error code
   */
  retCode init();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   *
   * @return     true if successful.
   */
  bool readParameters();

  /*!
   * ROS topic callback method.
   *
   * @param[in]  img_msg  The image message (to infer)
   */
  void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);

  //! ROS node handle.
  ros::NodeHandle& node_handle_;

  //! ROS topic subscribers and publishers.
  image_transport::ImageTransport it_;
  image_transport::Subscriber img_subscriber_;
  image_transport::Publisher bgr_publisher_;
  image_transport::Publisher mask_publisher_;
  image_transport::Publisher color_mask_publisher_;

  //! ROS topic names to subscribe to.
  std::string img_subscriber_topic_;
  std::string bgr_publisher_topic_;
  std::string mask_publisher_topic_;
  std::string color_mask_publisher_topic_;

  //! CNN related stuff
  Net* net_;
  std::string model_path_;
  std::string pb_model_path_;
  std::string model_;
  bool verbose_;
  std::string device_;
  std::string backend_;
  YAML::Node cfg_train_;
  YAML::Node cfg_net_;
  YAML::Node cfg_data_;
  YAML::Node cfg_nodes_;
};

} /* namespace */
