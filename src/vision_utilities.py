#!/usr/bin/env python
import os
import sys

import rospy
import rosservice

import constants
from config import VisionModuleConfiguration, parse_config
from services import initialize_services
from utils import ConsoleFormatter


class VisionUtilities:
    with_pepper: bool = False
    services_module = None
    main_camera = constants.PEPPER_FRONT_CAMERA
    config: VisionModuleConfiguration

    def __init__(self, config: VisionModuleConfiguration) -> None:
        self.config = config
        rospy.init_node(constants.NODE_NAME)

        self.using_pepper = (
            self.config.with_pepper
            and constants.VISION_TOOLS_SERVICE in rosservice.get_service_list()
        )
        if self.using_pepper:
            rospy.wait_for_service(constants.VISION_TOOLS_SERVICE)
        else:
            self.main_camera = constants.LOCAL_FRONT_CAMERA

        self.services_module = initialize_services(
            self.main_camera, self.config, self.config.ia
        )


if __name__ == "__main__":
    print(ConsoleFormatter.okgreen("Setting UP Vision utilities"))
    configuration = parse_config(sys.argv[1:])

    print(ConsoleFormatter.okblue("Using the following configuration:"))
    print("Configuration[" + str(configuration) + "]")

    vision_utilities = VisionUtilities(configuration)

    if vision_utilities.using_pepper:
        print(ConsoleFormatter.okgreen("-- Using remote Ros Master (Pepper)"))
    else:
        print(ConsoleFormatter.okgreen("-- Using local Ros Master"))
    rospy.spin()
