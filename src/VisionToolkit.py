#!/usr/bin/env python3
 
import rospy
import rosservice

import subprocess
import time

import constants

from common.ConsoleFormatter import ConsoleFormatter
from services.Initializer import initialize
from sensor_msgs.msg import Image


class VisionToolkit:
    is_ros_available: bool = False
    services_module = None
    main_camera = constants.PEPPER_FRONT_CAMERA

    def __init__(self) -> None:
        self.is_ros_available = False

        try:
            available_services = rosservice.get_service_list()
            self.is_ros_available = True
        except:
            self.is_ros_available = False

        if self.is_ros_available and constants.VISION_TOOLS_SERVICE in available_services:
            rospy.wait_for_service()
        else:
            subprocess.Popen("roscore")
            time.sleep(1)
            self.main_camera = constants.LOCAL_FRONT_CAMERA

        rospy.init_node("vision_toolkit")
        self.services_module = initialize(self.main_camera)

    
    def camera_subscriber(self):
        self.services_module



if __name__ == "__main__":
    print(ConsoleFormatter.okgreen("Setting UP Vision toolkit"))
    vision_toolkit = VisionToolkit()

    if vision_toolkit.is_ros_available:
        print(ConsoleFormatter.okgreen("-- Using remote Ros Master"))
    else:
        print(ConsoleFormatter.okgreen("-- Using local Ros Master "))
    rospy.spin()
