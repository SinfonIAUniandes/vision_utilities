from typing import Optional

import cv2
import rospy
from cv2.typing import MatLike
from perception_msgs.srv import read_qr_srv, read_qr_srvRequest

import constants
from utils.camera_topic import CameraTopic


class QrCodeScanner:
    def __init__(self, camera: str):
        rospy.Service(constants.SERVICE_READ_QR, read_qr_srv, self.callback)
        self.camera = CameraTopic(camera)
        self.detector = cv2.QRCodeDetector()

    def detect_qr(self, image: MatLike) -> Optional[str]:
        try:
            value, _, _ = self.detector.detectAndDecode(image)
            return value if value else None
        except:
            return None

    def callback(self, request: read_qr_srvRequest):
        result = self.camera.process_until(
            processor=self.detect_qr,
            timeout=request.timeout,
        )
        return result if result else ""
