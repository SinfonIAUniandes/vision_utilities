# Ros NODE Configuration

from pathlib import Path


NODE_NAME = "vision_utilities"

# Used services

VISION_TOOLS_SERVICE = "/robot_toolkit/vision_tools_srv"

# Models data

MODELS_FOLDER = Path(__file__).parent / "../models"

# Cameras

PEPPER_FRONT_CAMERA = "/robot_toolkit_node/camera/front/image_raw"
LOCAL_FRONT_CAMERA = "/camera/image_raw"

FRONT_CAMERA_NAME = "front_camera"

# Provided Services

SERVICE_READ_QR = "/vision_utilities/recognition/read_qr_srv"
SERVICE_DETECT_CHESS = "/vision_utilities/recognition/chess_srv"
SERVICE_DETECT_COCO_OBJECTS = "/vision_utilities/recognition/coco_objects_srv"
SERVICE_DETECT_FACE_LANDMARKS = "/vision_utilities/recognition/face_landmarks_srv"
SERVICE_DETECT_POSE_LANDMARKS = "/vision_utilities/recognition/pose_srv"
SERVICE_READ_IMAGE = "/vision_utilities/recognition/read_image_srv"
SERVICE_VLM = "/vision_utilities/recognition/vlm_srv"

# Topics

TOPIC_FACE_LANDMARKS = "/vision_utilities/recognition/face_landmarks_image"
TOPIC_POSE_LANDMARKS = "/vision_utilities/recognition/pose_landmarks_image"
TOPIC_COCO_DETECTIONS = "/vision_utilities/recognition/coco_detections_image"