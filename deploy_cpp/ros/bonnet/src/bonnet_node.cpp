#include <ros/ros.h>
#include "bonnet_handler.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "bonnet_node");
  ros::NodeHandle nodeHandle("~");

  bonnet::netHandler netH(nodeHandle);
  bonnet::retCode status = netH.init();
  if (status != bonnet::CNN_OK) {
    ROS_ERROR("SOMETHING WENT WRONG INITIALIZING CNN. EXITING");
    return 1;
  }

  ros::spin();
  return 0;
}
