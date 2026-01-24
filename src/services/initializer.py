import rospy
from robot_toolkit_msgs.msg import vision_tools_msg
from robot_toolkit_msgs.srv import vision_tools_srv

import constants
from config import VisionModuleConfiguration
from utils import ConsoleFormatter

from .mediapipe.hands_service import HandsService
from .mediapipe.pose_service import PoseService
from .OWL.nanoowl_detection_websocket import NanoOWLObjectDetectionService
from .qrcode_detection.QrCodeScanner import QrCodeScanner
from .vlm.img_description import VLMService
from .YOLO_based.coco_detection import COCOObjectDetectionService


def init_cameras():
    start_perception_message()


def start_perception_message():
    # Start front Camera
    vision_message = vision_tools_msg()
    vision_message.camera_name = constants.FRONT_CAMERA_NAME
    vision_message.command = "custom"
    vision_message.resolution = 1
    vision_message.frame_rate = 20
    vision_message.color_space = 11
    vision_tools_service(vision_message)


def vision_tools_service(msg):
    """
    Enables the vision Tools service from the toolkit of the robot.
    """
    print(ConsoleFormatter.warning("Waiting for vision tools service"))
    rospy.wait_for_service(constants.VISION_TOOLS_SERVICE)
    try:
        vision = rospy.ServiceProxy(constants.VISION_TOOLS_SERVICE, vision_tools_srv)
        vision(msg)
        print(ConsoleFormatter.okblue("Vision tools service connected!"))
    except rospy.ServiceException as e:
        print(ConsoleFormatter.error("Vision call failed"))


def initialize(camera: str, config: VisionModuleConfiguration, enable_ia: bool = False):
    if config.start_cameras:
        init_cameras()

    QrCodeScanner(camera)
    NanoOWLObjectDetectionService(camera, config)
    VLMService(camera, config.llm_mode, config.vlm_model, config.vlm_max_tokens)

    # TORCH DEPENDENT SERVICES
    if enable_ia:
        from .chess_detection.ChessDetection import ChessDetection
        from .mediapipe.face_landmark_service import FaceLandmarkService

        ChessDetection(camera)
        FaceLandmarkService(camera, config)
        PoseService(camera, config)
        HandsService(camera, config)
        COCOObjectDetectionService(camera, config)
