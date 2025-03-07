# Ros NODE Configuration

NODE_NAME = "vision_utilities"

# Used services

VISION_TOOLS_SERVICE = "/robot_toolkit/vision_tools_srv"

# Models data

MODELS_FOLDER = "./models/"

# Cameras

PEPPER_FRONT_CAMERA = "/robot_toolkit_node/camera/front/image_raw"
LOCAL_FRONT_CAMERA = "/camera/image_raw"

FRONT_CAMERA_NAME = "front_camera"

# Provided Services

SERVICE_READ_QR = "/vision_utilities/recognition/read_qr_srv"
SERVICE_DETECT_CHESS = "/vision_utilities/recognition/chess_srv"
