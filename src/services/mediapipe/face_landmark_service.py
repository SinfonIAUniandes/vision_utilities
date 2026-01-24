import cv2
import mediapipe as mp
import numpy as np
import rospy
from cv2.typing import MatLike
from cv_bridge import CvBridge
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from perception_msgs.msg import Polygon
from perception_msgs.srv import (
    ToggleDetectionTopic,
    ToggleDetectionTopicRequest,
    ToggleDetectionTopicResponse,
)

import constants
from config import VisionModuleConfiguration
from utils import models_manager
from utils.camera_topic import CameraTopic


class FaceLandmarkService:
    bridge = CvBridge()
    active = False

    def __init__(self, camera: str, config: VisionModuleConfiguration):
        self.model_asset_path = models_manager.get_mediapipe_path("face_landmarker")
        self.detector = self.initialize_face_landmarker()
        self.config = config
        self.service = rospy.Service(
            constants.SERVICE_DETECT_FACE_LANDMARKS,
            ToggleDetectionTopic,
            self.handle_face_landmark_detection,
        )
        self.image_pub = None
        if "face_landmarks" in config.publish_visualizations:
            self.image_pub = rospy.Publisher(
                constants.TOPIC_FACE_LANDMARKS, Polygon, queue_size=10
            )
            print(self.image_pub)
        self.camera = CameraTopic(camera)
        self.sid = None

    def initialize_face_landmarker(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_asset_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5,
        )
        return FaceLandmarker.create_from_options(options)

    def get_landmark_polygons(self, detection_result):
        all_polygons = []
        face_landmarks_list = detection_result.face_landmarks
        connections = (
            mp.solutions.face_mesh.FACEMESH_TESSELATION
            | mp.solutions.face_mesh.FACEMESH_CONTOURS
            | mp.solutions.face_mesh.FACEMESH_IRISES
        )

        for face_landmarks in face_landmarks_list:
            for connection in connections:
                start_idx, end_idx = connection
                start_lm = face_landmarks[start_idx]
                end_lm = face_landmarks[end_idx]
                polygon = [start_lm.x, start_lm.y, end_lm.x, end_lm.y]
                all_polygons.append(polygon)

        return np.array(all_polygons).flatten().tolist()

    def camera_subscriber(self, image: MatLike):
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        polygons = self.get_landmark_polygons(detection_result)

        if polygons and self.image_pub is not None:
            response = Polygon()
            response.label = "face"
            response.polygon = polygons
            self.image_pub.publish(response)

    def handle_face_landmark_detection(self, req: ToggleDetectionTopicRequest):
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
