import rospy

from perception_msgs.srv import (
    detect_chess_srv,
    detect_chess_srvRequest,
    detect_chess_srvResponse,
)
from sensor_msgs.msg import Image
from robot_toolkit_msgs.msg import vision_tools_msg
from robot_toolkit_msgs.srv import vision_tools_srv
import constants

import cv2
from cv_bridge import CvBridge
import time

from services.chess_detection import board_detection, pieces_detection


class ChessDetection:
    bridge = CvBridge()
    image = None

    def __init__(self, camera: str):
        rospy.Service(constants.SERVICE_DETECT_CHESS, detect_chess_srv, self.callback)
        rospy.Subscriber(camera, Image, self.camera_subscriber)

    def camera_subscriber(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def callback(self, request: detect_chess_srvRequest):
        response = detect_chess_srvResponse()
        response.board = []
        response.pieces = []
        response.fen = ""


        start = time.time()
        to_process = self.image

        while time.time() - start < request.timeout:
            if self.image is not None:
                to_process = self.image
                break
        if to_process is None:
            return response
        
        to_process = cv2.resize(to_process, (640, 640))

        start_time = time.time()

        board_corners = board_detection.get_corners(to_process)

        end_time = time.time()
        print("Took %.2fs" % (end_time - start_time))

        response.board = board_corners
        
        pieces = pieces_detection.get_predictions(to_process)


        return response
