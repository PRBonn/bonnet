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
