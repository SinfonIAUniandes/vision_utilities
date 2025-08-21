# Ros NODE Configuration

from pathlib import Path


NODE_NAME = "vision_utilities"
WEBCAM_PUBLISHER_NAME = "vision_utilities_webcam_publisher"
POLYGON_RENDERING_NAME = "vision_utilities_polygon_rendering"

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
SERVICE_DETECT_FACE_LANDMARKS = "/vision_utilities/recognition/face_landmarks_srv"
SERVICE_RENDER_POLYGON_TOPIC = "/vision_utilities/rendering/visualize_polygon_topic_srv"

# Topics

TOPIC_FACE_LANDMARKS = "/vision_utilities/recognition/face_landmarks_polygons"


TOPIC_POLYGON_RENDERER = "/vision_utilities/rendering/polygons"