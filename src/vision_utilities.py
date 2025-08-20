#!/usr/bin/env python
import rospy
import rosservice
import os

import sys

import constants

from utils import ConsoleFormatter
from services import initialize_services

from config import VisionModuleConfiguration, parse_config


class VisionUtilities:
    with_pepper: bool = False
    services_module = None
    main_camera = constants.PEPPER_FRONT_CAMERA
    config: VisionModuleConfiguration

    def __init__(self, config: VisionModuleConfiguration) -> None:
        self.config = config
        rospy.init_node(constants.NODE_NAME)

        if (
            self.config.with_pepper
            and constants.VISION_TOOLS_SERVICE in rosservice.get_service_list()
        ):
            rospy.wait_for_service(constants.VISION_TOOLS_SERVICE)
        else:
            self.main_camera = constants.LOCAL_FRONT_CAMERA

        self.services_module = initialize_services(self.main_camera, self.config, self.config.ia)


if __name__ == "__main__":
    print(ConsoleFormatter.okgreen("Setting UP Vision utilities"))
    configuration = parse_config(sys.argv[1:])

    print(ConsoleFormatter.okblue("Using the following configuration:"))
    print("Configuration[" + str(configuration) + "]")

    vision_utilities = VisionUtilities(configuration)

    if configuration.with_pepper:
        print(ConsoleFormatter.okgreen("-- Using remote Ros Master"))
    else:
        print(ConsoleFormatter.okgreen("-- Using local Ros Master "))
    rospy.spin()
