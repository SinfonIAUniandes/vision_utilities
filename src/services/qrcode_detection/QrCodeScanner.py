import rospy
import time

from perception_msgs.srv import read_qr_srv, read_qr_srvRequest
import constants

import cv2
from cv_bridge import CvBridge
from utils.camera_topic import CameraTopic
from cv2.types import MatLike


class QrCodeScanner:
    bridge = CvBridge()
    detected = None

    def __init__(self, camera: str):
        rospy.Service(constants.SERVICE_READ_QR, read_qr_srv, self.callback)
        self.camera = CameraTopic(camera)
        self.detected = None

    def camera_subscriber(self, image: MatLike):
        detector = cv2.QRCodeDetector()
        try:
            value, _, _ = detector.detectAndDecode(image)
            if value != "":
                self.detected = value
        except:
            pass

    def callback(self, request: read_qr_srvRequest):
        sid = self.camera.subscribe(self.camera_subscriber, wait_turns=1)
        start = time.time()

        while time.time() - start < request.timeout:
            if self.detected is not None:
                break
            time.sleep(0.1)
        self.camera.unsubscribe(sid)
        return self.detected if self.detected is not None else ""