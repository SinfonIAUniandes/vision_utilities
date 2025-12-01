from common import models_manager
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.srv import FaceLandmarkDetectionRequest, FaceLandmarkDetectionResponse, FaceLandmarkDetection
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from pathlib import Path
import rospy
import cv2
import mediapipe as mp
import numpy as np
import constants

class PoseService:
    bridge = CvBridge()
    image = None
    active = False

    def __init__(self, camera: str):
        self.active = False
        self.model_asset_path = models_manager.get_mediapipe_path("pose_landmarker")
        self.image_pub = rospy.Publisher(constants.TOPIC_POSE_LANDMARKS, Image, queue_size=10)
        self.service = rospy.Service(constants.SERVICE_DETECT_POSE_LANDMARKS, FaceLandmarkDetection, self.handle_pose_detection)
        mp_pose = mp.solutions.pose

        self.drawing_styles = mp.solutions.drawing_styles
        self.drawing_utils = mp.solutions.drawing_utils
        self.pose_connections = mp.solutions.pose
        self.detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        rospy.Subscriber(camera, Image, self.camera_subscriber)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        rgb_image.flags.writeable = True
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if detection_result.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                rgb_image,
                detection_result.pose_landmarks,
                self.pose_connections.POSE_CONNECTIONS,
                self.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                self.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )
        annotated_image = np.copy(rgb_image)

        return annotated_image

    def camera_subscriber(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        if not self.active:
            return

        if self.image is None:
            rospy.logerr("No image received from camera.")
            return

        rgb_frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        detection_result = self.detector.process(rgb_frame)

        # Process the detection result - Visualize it
        annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)

        annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.image_pub.publish(annotated_image_msg)

    def handle_pose_detection(self, req: FaceLandmarkDetectionRequest):
        response = FaceLandmarkDetectionResponse()
        if req.state:
            self.active = True
        else:
            self.active = False
            
        response.state = "Active:" + str(self.active)
        return response