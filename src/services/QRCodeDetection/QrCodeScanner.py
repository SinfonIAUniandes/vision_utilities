import rospy
import time

from perception_msgs.srv import read_qr_srv, read_qr_srvRequest
from sensor_msgs.msg import Image
from robot_toolkit_msgs.msg import vision_tools_msg
from robot_toolkit_msgs.srv import vision_tools_srv

import cv2
from cv_bridge import CvBridge

class QrCodeScanner:
    bridge = CvBridge()
    image = None


    def __init__(self, camera: str):
        rospy.Service("vision_utilities/read_qr_srv", read_qr_srv, self.callback)
        rospy.Subscriber(camera, Image, self.camera_subscriber)
        self.start_perception_message()

    def start_perception_message(self):
        #Start front Camera
        visionMessage = vision_tools_msg()
        visionMessage.camera_name = "front_camera"
        visionMessage.command = "custom"
        visionMessage.resolution = 1
        visionMessage.frame_rate = 20
        visionMessage.color_space = 11
        self.visionToolsService(visionMessage)

    def visionToolsService(self,msg):
        """
        Enables the vision Tools service from the toolkit of the robot.
        """
        print("Waiting for vision tools service")
        rospy.wait_for_service('/robot_toolkit/vision_tools_srv')
        try:
            vision = rospy.ServiceProxy('/robot_toolkit/vision_tools_srv', vision_tools_srv)
            visionService = vision(msg)
            print("Vision tools service connected!")
        except rospy.ServiceException as e:
            print("Vision call failed")


    def camera_subscriber(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def callback(self, request: read_qr_srvRequest):
        start = time.time()

        while (time.time() - start < request.timeout):
            try:
                detector = cv2.QRCodeDetector()

                value, _,_ = detector.detectAndDecode(self.image)

                if value != "":
                    return value
            except:
                continue
        return ""