#!/usr/bin/env python3
 
import rospy
import rosservice

import sys

import constants

from common.ConsoleFormatter import ConsoleFormatter
from services.Initializer import initialize

from config import VisionConfiguration, parse_config


class VisionToolkit:
    with_pepper: bool = False
    services_module = None
    main_camera = constants.PEPPER_FRONT_CAMERA
    config: VisionConfiguration

    def __init__(self, config: VisionConfiguration) -> None:
        self.config = config
        self.with_pepper = False

        if self.with_pepper and constants.VISION_TOOLS_SERVICE in rosservice.get_service_list():
            rospy.wait_for_service(constants.VISION_TOOLS_SERVICE)
        else:
            self.main_camera = constants.LOCAL_FRONT_CAMERA

        rospy.init_node(constants.NODE_NAME)
        self.services_module = initialize(self.main_camera, self.config)

    
    def camera_subscriber(self):
        self.services_module


if __name__ == "__main__":
    print(ConsoleFormatter.okgreen("Setting UP Vision utilities"))
    configuration = parse_config(sys.argv)
    vision_toolkit = VisionToolkit(configuration)

    if vision_toolkit.with_pepper:
        print(ConsoleFormatter.okgreen("-- Using remote Ros Master"))
    else:
        print(ConsoleFormatter.okgreen("-- Using local Ros Master "))
    rospy.spin()
