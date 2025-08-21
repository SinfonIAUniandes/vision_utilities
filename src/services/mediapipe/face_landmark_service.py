from utils import models_manager
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.msg import Polygon
from perception_msgs.srv import FaceLandmarkDetectionRequest, FaceLandmarkDetectionResponse, FaceLandmarkDetection
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from pathlib import Path
import rospy
import cv2
from cv2.typing import MatLike
import mediapipe as mp
import numpy as np
import constants
from utils.camera_topic import CameraTopic

class FaceLandmarkService:
    bridge = CvBridge()
    active = False
    frames_interval = 3

    def __init__(self, camera: str):
        self.active = False
        self.model_asset_path = models_manager.get_mediapipe_path("face_landmarker")
        self.detector = self.initialize_face_landmarker()
        self.image_pub = rospy.Publisher(constants.TOPIC_FACE_LANDMARKS, Polygon, queue_size=10)
        self.service = rospy.Service(constants.SERVICE_DETECT_FACE_LANDMARKS, FaceLandmarkDetection, self.handle_face_landmark_detection)
        self.camera = CameraTopic(camera)

    def initialize_face_landmarker(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_asset_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5,
        )
        return FaceLandmarker.create_from_options(options)
    

    def get_landmark_polygons(self, detection_result):
        """
        Extracts face landmark connections as a list of polygons (lines).

        Args:
            detection_result: The result from a MediaPipe face landmarker detection.

        Returns:
            A list of polygons, where each polygon is a list of floats
            representing a line segment, e.g., [x1, y1, x2, y2].
        """
        all_polygons = []
        face_landmarks_list = detection_result.face_landmarks

        # Combine all connection sets from MediaPipe Face Mesh
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
                
                # A line is a polygon of two points [x1, y1, x2, y2]
                polygon = [start_lm.x, start_lm.y, end_lm.x, end_lm.y]
                all_polygons.append(polygon)

        return np.array(all_polygons).flatten().tolist()
    

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        drawing_styles = mp.solutions.drawing_styles
        drawing_utils = mp.solutions.drawing_utils

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

            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )

            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image
    
    def get_bounding_rect(self, landmarks, image_shape):
        """
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
        Simple rules based on blendshape scores:
        - Smile if both mouth corners high
        - Surprise if jaw dropped + brows raised
        - Frown if brows lowered
        - Otherwise neutral
        """
        scores = {cat.category_name: cat.score for cat in blendshapes}

        if (
            scores.get("mouthSmileLeft", 0) > threshSmile
            and scores.get("mouthSmileRight", 0) > threshSmile
        ):
            return " Sonriendo"
        if scores.get("jawOpen", 0) > threshBrow and (
            scores.get("browOuterUpLeft", 0) > threshBrow
            or scores.get("browOuterUpRight", 0) > threshBrow
        ):
            return "Sorprendido"
        if (
            scores.get("browDownLeft", 0) > threshBrow
            and scores.get("browDownRight", 0) > threshBrow
        ):
            return " Bravo"
        return " Neutral"



    def camera_subscriber(self, image: MatLike):
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        polygons = self.get_landmark_polygons(detection_result)

        response = Polygon()
        response.label = "face"
        response.polygon = polygons

        self.image_pub.publish(response)
        return 

        # DEBUG: print top 5 blendshapes for the first detected face
        if detection_result.face_blendshapes:
            bs0 = detection_result.face_blendshapes[0]
            # sort by descending score
            top5 = sorted(bs0, key=lambda c: -c.score)[:5]
            print(
                "Top blendshapes:", [(c.category_name, round(c.score, 3)) for c in top5]
            )

        # Check for face presence
        if detection_result.face_landmarks:
            num = len(detection_result.face_landmarks)
            print(f"[INFO] Detected {num} face(s)")
            # (optional) overlay text on the frame:
            cv2.putText(
                image,
                f"Faces: {num}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        else:
            print("[INFO] No faces detected")
            # (optional) overlay text on the frame:
            cv2.putText(
                image,
                "No face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,)
           # Process the detection result - Visualize it
        annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)

        # 1) Convert the annotated image to BGR for OpenCV
        frame_out = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # 2) Overlay the face-count text onto frame_out
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
        # 3) Draw boxes + expression labels
        for i, landmarks in enumerate(detection_result.face_landmarks):
            # a) Bounding box
            x, y, w_box, h_box = self.get_bounding_rect(landmarks, frame_out.shape)
            cv2.rectangle(frame_out, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

            # b) Classify expression
            expr = self.classify_expression(detection_result.face_blendshapes[i])

            # c) Compute text position (mid-forehead)
            h_img, w_img = frame_out.shape[:2]
            eye_l = landmarks[33]  # right eye inner corner
            eye_r = landmarks[263]  # left eye inner corner
            text_x = int(((eye_l.x + eye_r.x) / 2) * w_img)
            text_y = int(min(eye_l.y, eye_r.y) * h_img) - 10

            # d) Overlay the label
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


        
        annotated_image_msg = self.bridge.cv2_to_imgmsg(frame_out, encoding="bgr8")

    def handle_face_landmark_detection(self, req: FaceLandmarkDetectionRequest):
        response = FaceLandmarkDetectionResponse()

        if self.active == req.state and req.frames_interval == self.frames_interval:
            response.state = "Already " + ("active" if self.active else "inactive")
            return response
        
        if self.active:
            self.active = False
            self.camera.unsubscribe(self.sid)
            response.state = "Deactivated"

        self.frames_interval = req.frames_interval


        if req.state:
            self.active = True
            self.sid = self.camera.subscribe(self.camera_subscriber, wait_turns=req.frames_interval)
            response.state = "Activated"
        else:
            self.active = False
            self.camera.unsubscribe(self.sid)
            response.state = "Deactivated"
        
        return response