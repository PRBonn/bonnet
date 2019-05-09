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
// STD
#include <unistd.h>
#include <string>

// net stuff
#include "bonnet_handler.hpp"

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
  alpha_blend_publisher_ = it_.advertise(alpha_blend_publisher_topic_, 1);

  ROS_INFO("Successfully launched node.");
}

/*!
 * @brief      Initialize the Handler
 *
 * @return     Error code
 */
retCode netHandler::init() {
  // before doing anything, make sure we have a slash at the end of path
  path_ += "/";

  try {
    net_ =
        std::unique_ptr<Bonnet>(new Bonnet(path_, backend_, device_, verbose_));
  } catch (const std::invalid_argument& e) {
    std::cerr << "Unable to create network. " << std::endl
              << e.what() << std::endl;
    return CNN_FATAL;
  } catch (const std::runtime_error& e) {
    std::cerr << "Unable to init. network. " << std::endl
              << e.what() << std::endl;
    return CNN_FATAL;
  }

  return CNN_OK;
}

/*!
 * Destructor.
 */
netHandler::~netHandler() {}

bool netHandler::readParameters() {
  if (!node_handle_.getParam("image_topic", img_subscriber_topic_) ||
      !node_handle_.getParam("bgr_topic", bgr_publisher_topic_) ||
      !node_handle_.getParam("mask_topic", mask_publisher_topic_) ||
      !node_handle_.getParam("color_mask_topic", color_mask_publisher_topic_) ||
      !node_handle_.getParam("alpha_blend_topic",
                             alpha_blend_publisher_topic_) ||
      !node_handle_.getParam("model_path", path_) ||
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

  // Send the alpha blend
  cv::Mat alpha_blend(mask_bgr.rows, mask_bgr.cols, CV_8UC3);
  net_->blend(cv_img_bgr, 1, mask_bgr, 0.5, alpha_blend, verbose_);
  sensor_msgs::ImagePtr blend_msg =
      cv_bridge::CvImage(img_msg->header, "bgr8", alpha_blend).toImageMsg();
  alpha_blend_publisher_.publish(blend_msg);

  // Echo the image as bgr
  sensor_msgs::ImagePtr bgr_msg =
      cv_bridge::CvImage(img_msg->header, "bgr8", cv_img_bgr).toImageMsg();
  bgr_publisher_.publish(bgr_msg);
}

}  // namespace bonnet
