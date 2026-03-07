# Vision Utilities

**Vision Utilities** is a **ROS Noetic** package and a submodule of the **Perception Module** in the **SinfonIA** workspace. It provides essential vision-related functionalities and services for the perception system, including support for QR code recognition, object detection, pose estimation, and more.

## Installation & Setup

To use this package, you must clone it into a ROS Noetic workspace and install the specific Python dependencies based on your use case.

### 1. Clone the Repository

Navigate to the `src` folder of your catkin workspace and clone the repository using SSH:

```bash
cd ~/catkin_ws/src
git clone git@github.com:SinfonIAUniandes/vision_utilities.git

```

### 2. Install Dependencies

This repo contains three different requirement files. Choose the one that fits your needs and install it via `pip`:

* **All Dependencies:** Installs every requirement for all features.
```bash
pip install -r all_requirements.txt

```


* **AI Features:** Includes MediaPipe, Ultralytics, and DEAP for vision tasks.
```bash
pip install -r requirements_ai.txt

```


* **Basic Usage:** Only core functionalities (no AI/Heavy models).
```bash
pip install -r requirements_basic.txt

```



### 3. Build the Workspace

Return to the root of your workspace and build the package:

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash

```

---

## External Camera Setup (`usb_cam`)

Instead of a custom publisher, we use the standard `usb_cam` package to stream video from local webcams.

### Installation

Install the package via the official ROS binaries:

```bash
sudo apt update
sudo apt install ros-noetic-usb-cam

```

### Usage

To start your webcam and publish frames to the ROS network:

```bash
# Basic start (usually /dev/video0)
rosrun usb_cam usb_cam_node

# Start with specific parameters (e.g., pixel format or device)
rosrun usb_cam usb_cam_node _video_device:="/dev/video0" _pixel_format:="yuyv"

```

The images will be published by default to `/usb_cam/image_raw`.

---

## Overview

The Vision Utilities submodule offers ROS services tailored for perception tasks. The configuration can be tailored based on the environment setup, particularly for robots such as **Pepper**.

### ROS Services

| Service | Description |
| --- | --- |
| `/vision_utilities/recognition/read_qr_srv` | Reads and interprets QR codes |
| `/vision_utilities/recognition/coco_objects_srv` | Toggles YOLO-based object detection |
| `/vision_utilities/recognition/owl_objects_srv` | Toggles NanoOWL object detection |
| `/vision_utilities/recognition/vlm_srv` | Vision-Language Model service (Ollama/OpenAI) |
| `/vision_utilities/recognition/face_landmarks_srv` | Toggles face landmark detection (requires `--ia`) |
| `/vision_utilities/recognition/pose_srv` | Toggles pose landmark detection (requires `--ia`) |

### ROS Topics

| Topic | Message Type | Description |
| --- | --- | --- |
| `/vision_utilities/recognition/coco_detections_image` | `sensor_msgs/Image` | Annotated image with COCO detections |
| `/vision_utilities/recognition/pose_landmarks_image` | `sensor_msgs/Image` | Annotated image with pose landmarks |
| `/vision_utilities/rendering/polygons` | `sensor_msgs/Image` | Rendered polygon overlays |

---

## Configuration

### Command-Line Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--with_pepper` | bool | `False` | Enables compatibility with the Pepper robot |
| `--start_cameras` | bool | `False` | Initializes the robot's cameras during startup |
| `--ia` | bool | `False` | Enables AI-dependent services (face, pose, hand) |
| `--llm_mode` | string | `"ollama"` | VLM backend: `"ollama"` or `"openai"` |

### Examples

**Running with a local webcam (via usb_cam):**

1. In terminal 1: `rosrun usb_cam usb_cam_node`
2. In terminal 2: `rosrun vision_utilities vision_utilities.py`

**Running for Pepper with AI features:**

```bash
rosrun vision_utilities vision_utilities.py --with_pepper --start_cameras --ia

```

## License

This repository is proprietary and is intended solely for use by **SinfonIA**. Unauthorized use, distribution, or modification is prohibited.
