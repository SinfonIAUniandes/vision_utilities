import rospy
import time

from perception_msgs.srv import read_qr_srv, read_qr_srvRequest
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge

class QrCodeScanner:
    bridge = CvBridge()
    image = None

    def __init__(self, camera: str):
        rospy.Service("vision_utilities/read_qr_srv", read_qr_srv, self.callback)
        rospy.Subscriber(camera, Image, self.camera_subscriber)

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