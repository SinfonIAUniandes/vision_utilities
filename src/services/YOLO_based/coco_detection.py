from common import models_manager
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.srv import FaceLandmarkDetectionRequest, FaceLandmarkDetectionResponse, FaceLandmarkDetection
import rospy
import cv2
import numpy as np
import constants

class COCOObjectDetectionService:
    bridge = CvBridge()
    image = None
    active = False

    def __init__(self, camera: str):
        self.active = False
        self.model = models_manager.get_yolo_model("yolo11s")
        self.image_pub = rospy.Publisher(constants.TOPIC_COCO_DETECTIONS, Image, queue_size=10)
        self.service = rospy.Service(constants.SERVICE_DETECT_COCO_OBJECTS, FaceLandmarkDetection, self.handle_coco_object_detection)
        rospy.Subscriber(camera, Image, self.camera_subscriber)

    def handle_coco_object_detection(self, req: FaceLandmarkDetectionRequest):
        response = FaceLandmarkDetectionResponse()
        if req.state:
            self.active = True
        else:
            self.active = False
            
        response.state = "Active:" + str(self.active)
        return response

    def camera_subscriber(self, msg: Image):
        if not self.active:
            return

        # Convertir la imagen ROS a OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Realizar la inferencia con el modelo YOLO
        results = self.model(cv_image)

        # Anotar la imagen con las detecciones
        annotated_image = results[0].plot()

        # Publicar la imagen anotada
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        self.image_pub.publish(annotated_msg)