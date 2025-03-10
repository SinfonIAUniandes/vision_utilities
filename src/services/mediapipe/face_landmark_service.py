from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.srv import FaceLandmarkDetection, FaceLandmarkDetectionResponse  # Update import
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from pathlib import Path
import rospy
import cv2
import mediapipe as mp
import numpy as np

class FaceLandmarkService:
    bridge = CvBridge()
    image = None

    def __init__(self, camera: str):
        self.model_asset_path = Path("/home/emilio/Documents/Sinfonia/Face_Recognition/face_landmarker_v2_with_blendshapes.task")
        self.detector = self.initialize_face_landmarker()
        self.image_pub = rospy.Publisher("face_landmarks_image", Image, queue_size=10)
        self.service = rospy.Service('face_landmark_detection', FaceLandmarkDetection, self.handle_face_landmark_detection)
        rospy.Subscriber(camera, Image, self.camera_subscriber)

    def initialize_face_landmarker(self):
        base_options = vision.BaseOptions(model_asset_path=str(self.model_asset_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5,
        )
        return vision.FaceLandmarker.create_from_options(options)

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

    def camera_subscriber(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def handle_face_landmark_detection(self, req):
        if self.image is None:
            rospy.logerr("No image received from camera.")
            return FaceLandmarkDetectionResponse(annotated_image=Image())

        rgb_frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
        annotated_image_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR), encoding="bgr8")
        self.image_pub.publish(annotated_image_msg)
        return FaceLandmarkDetectionResponse(annotated_image=annotated_image_msg)