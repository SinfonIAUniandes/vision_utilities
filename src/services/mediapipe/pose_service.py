import cv2
import mediapipe as mp
import numpy as np
import rospy
from cv2.typing import MatLike
from cv_bridge import CvBridge
from perception_msgs.srv import (
    ToggleDetectionTopic,
    ToggleDetectionTopicRequest,
    ToggleDetectionTopicResponse,
)
from sensor_msgs.msg import Image

import constants
from common import models_manager
from utils.camera_topic import CameraTopic


class PoseService:
    bridge = CvBridge()
    active = False

    def __init__(self, camera: str):
        self.model_asset_path = models_manager.get_mediapipe_path("pose_landmarker")
        self.image_pub = rospy.Publisher(
            constants.TOPIC_POSE_LANDMARKS, Image, queue_size=10
        )
        self.service = rospy.Service(
            constants.SERVICE_DETECT_POSE_LANDMARKS,
            ToggleDetectionTopic,
            self.handle_pose_detection,
        )

        mp_pose = mp.solutions.pose
        self.drawing_styles = mp.solutions.drawing_styles
        self.drawing_utils = mp.solutions.drawing_utils
        self.pose_connections = mp.solutions.pose
        self.detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.camera = CameraTopic(camera)
        self.sid = None

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        rgb_image.flags.writeable = True
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if detection_result.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                rgb_image,
                detection_result.pose_landmarks,
                self.pose_connections.POSE_CONNECTIONS,
                self.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                self.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                ),
            )
        return np.copy(rgb_image)

    def camera_subscriber(self, image: MatLike):
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_result = self.detector.process(rgb_frame)
        annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
        annotated_image_msg = self.bridge.cv2_to_imgmsg(
            annotated_image, encoding="bgr8"
        )
        self.image_pub.publish(annotated_image_msg)

    def handle_pose_detection(self, req: ToggleDetectionTopicRequest):
        response = ToggleDetectionTopicResponse()
        if req.state:
            self.active = True
            self.sid = self.camera.subscribe(
                self.camera_subscriber, wait_turns=req.frames_interval
            )
            response.state = "Activated"
        else:
            self.active = False
            if self.sid is not None:
                self.camera.unsubscribe(self.sid)
            response.state = "Deactivated"
        return response
