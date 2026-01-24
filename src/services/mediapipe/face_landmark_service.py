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
from sensor_msgs.msg import Image

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
        self.drawing_styles = mp.solutions.drawing_styles
        self.drawing_utils = mp.solutions.drawing_utils
        self.service = rospy.Service(
            constants.SERVICE_DETECT_FACE_LANDMARKS,
            ToggleDetectionTopic,
            self.handle_face_landmark_detection,
        )
        # Visualization publisher (Image)
        self.image_pub = None
        if "face_landmarks" in config.publish_visualizations:
            self.image_pub = rospy.Publisher(
                constants.TOPIC_FACE_LANDMARKS, Image, queue_size=10
            )
        # Polygon data publisher
        self.polygon_pub = None
        if "polygon_data" in config.publish_data:
            self.polygon_pub = rospy.Publisher(
                constants.TOPIC_POLYGON_RENDERER, Polygon, queue_size=10
            )
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

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """Draw face mesh landmarks on image (tesselation, contours, irises)."""
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        for face_landmarks in face_landmarks_list:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in face_landmarks
                ]
            )

            self.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            self.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_face_mesh_contours_style(),
            )

            self.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image

    def get_bounding_rect(self, landmarks, image_shape):
        """
        Get bounding box for face landmarks.
        landmarks: list of objects with .x and .y normalized to [0,1]
        image_shape: frame.shape (height, width, _)
        returns: (x, y, w, h) in pixel coords
        """
        h, w = image_shape[:2]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        return x_min, y_min, x_max - x_min, y_max - y_min

    def classify_expression(self, blendshapes, threshSmile=0.6, threshBrow=0.5):
        """
        Classify expression based on blendshape scores.
        Returns expression label.
        """
        scores = {cat.category_name: cat.score for cat in blendshapes}

        if (
            scores.get("mouthSmileLeft", 0) > threshSmile
            and scores.get("mouthSmileRight", 0) > threshSmile
        ):
            return "Sonriendo"
        if scores.get("jawOpen", 0) > threshBrow and (
            scores.get("browOuterUpLeft", 0) > threshBrow
            or scores.get("browOuterUpRight", 0) > threshBrow
        ):
            return "Sorprendido"
        if (
            scores.get("browDownLeft", 0) > threshBrow
            and scores.get("browDownRight", 0) > threshBrow
        ):
            return "Bravo"
        return "Neutral"

    def camera_subscriber(self, image: MatLike):
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        # Publish visualization if enabled
        if self.image_pub is not None:
            # Draw landmarks on image
            annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
            frame_out = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # Overlay face count
            if detection_result.face_landmarks:
                count = len(detection_result.face_landmarks)
                cv2.putText(
                    frame_out,
                    f"Faces: {count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame_out,
                    "No face",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Draw boxes and expression labels
            for i, landmarks in enumerate(detection_result.face_landmarks):
                # Bounding box
                x, y, w_box, h_box = self.get_bounding_rect(landmarks, frame_out.shape)
                cv2.rectangle(frame_out, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                # Expression classification
                expr = self.classify_expression(detection_result.face_blendshapes[i])

                # Compute text position (mid-forehead)
                h_img, w_img = frame_out.shape[:2]
                eye_l = landmarks[33]  # right eye inner corner
                eye_r = landmarks[263]  # left eye inner corner
                text_x = int(((eye_l.x + eye_r.x) / 2) * w_img)
                text_y = int(min(eye_l.y, eye_r.y) * h_img) - 10

                # Overlay expression label
                cv2.putText(
                    frame_out,
                    expr,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Publish visualization
            annotated_image_msg = self.bridge.cv2_to_imgmsg(frame_out, encoding="bgr8")
            self.image_pub.publish(annotated_image_msg)

        # Publish polygon data if enabled
        if self.polygon_pub is not None:
            polygons = self.get_landmark_polygons(detection_result)
            if polygons:
                response = Polygon()
                response.label = "face"
                response.polygon = polygons
                self.polygon_pub.publish(response)

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
