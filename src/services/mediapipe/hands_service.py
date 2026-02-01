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
from config import VisionModuleConfiguration
from utils import models_manager
from utils.camera_topic import CameraTopic


class HandsService:
    bridge = CvBridge()
    active = False

    def __init__(self, camera: str, config: VisionModuleConfiguration):
        self.model_asset_path = models_manager.get_mediapipe_path("hand_landmarker")
        self.config = config
        self.image_pub = None
        if "hand_landmarks" in config.publish_visualizations:
            self.image_pub = rospy.Publisher(
                constants.TOPIC_HAND_LANDMARKS, Image, queue_size=10
            )
        self.service = rospy.Service(
            constants.SERVICE_DETECT_HAND_LANDMARKS,
            ToggleDetectionTopic,
            self.handle_hand_detection,
        )

        mp_hands = mp.solutions.hands
        self.drawing_styles = mp.solutions.drawing_styles
        self.drawing_utils = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands
        self.detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.camera = CameraTopic(camera)
        self.sid = None

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        rgb_image.flags.writeable = True
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if detection_result.multi_hand_landmarks:
            for hand_landmarks in detection_result.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    rgb_image,
                    hand_landmarks,
                    self.hand_connections.HAND_CONNECTIONS,
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
        if self.image_pub is not None:
            annotated_image_msg = self.bridge.cv2_to_imgmsg(
                annotated_image, encoding="bgr8"
            )
            self.image_pub.publish(annotated_image_msg)

    def handle_hand_detection(self, req: ToggleDetectionTopicRequest):
        response = ToggleDetectionTopicResponse()
        if req.state:
            if self.active and self.sid is not None:
                self.camera.unsubscribe(self.sid)
            frames_interval = max(1, req.frames_interval)
            self.active = True
            self.sid = self.camera.subscribe(
                self.camera_subscriber, wait_turns=frames_interval
            )
            response.state = "Activated"
        else:
            self.active = False
            if self.sid is not None:
                self.camera.unsubscribe(self.sid)
                self.sid = None
            response.state = "Deactivated"
        return response
