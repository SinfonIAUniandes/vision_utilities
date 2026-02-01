# Vision Utilities

**Vision Utilities** is a submodule of the **Perception Module** in the **SinfonIA** workspace. It provides essential vision-related functionalities and services for the perception system, including support for QR code recognition, object detection, pose estimation, and more.

## Overview

The Vision Utilities submodule offers ROS services tailored for perception tasks, enabling efficient handling of various vision-based operations. The configuration can be tailored based on the environment setup, particularly for robots such as Pepper. The module parses configuration options at runtime, allowing flexible deployment scenarios.

### ROS Services

The following ROS services are provided:

| Service | Description |
|---------|-------------|
| `/vision_utilities/recognition/read_qr_srv` | Reads and interprets QR codes within the environment |
| `/vision_utilities/recognition/coco_objects_srv` | Toggles COCO object detection (YOLO-based) topic publishing |
| `/vision_utilities/recognition/owl_objects_srv` | Toggles NanoOWL object detection topic publishing |
| `/vision_utilities/recognition/owl_objects_srv_prompt` | Sets the detection prompt for NanoOWL |
| `/vision_utilities/recognition/vlm_srv` | Vision-Language Model service for image description |
| `/vision_utilities/recognition/chess_srv` | Detects chess board and pieces, returns FEN notation |
| `/vision_utilities/recognition/face_landmarks_srv` | Toggles face landmark detection (requires `--ia`) |
| `/vision_utilities/recognition/pose_srv` | Toggles pose landmark detection (requires `--ia`) |
| `/vision_utilities/recognition/hand_srv` | Toggles hand landmark detection (requires `--ia`) |
| `/vision_utilities/rendering/visualize_polygon_topic_srv` | Subscribes to polygon topics and renders them on camera feed |

### ROS Topics

Published topics for detection visualization:

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/vision_utilities/recognition/coco_detections_image` | `sensor_msgs/Image` | Annotated image with COCO detections |
| `/vision_utilities/recognition/owl_detections_image` | `sensor_msgs/Image` | Annotated image with OWL detections |
| `/vision_utilities/recognition/face_landmarks_image` | `perception_msgs/Polygon` | Face landmark polygons |
| `/vision_utilities/recognition/pose_landmarks_image` | `sensor_msgs/Image` | Annotated image with pose landmarks |
| `/vision_utilities/recognition/hand_landmarks_image` | `sensor_msgs/Image` | Annotated image with hand landmarks |
| `/vision_utilities/rendering/polygons` | `sensor_msgs/Image` | Rendered polygon overlays |

## Configuration

Vision Utilities allows configuration via command-line arguments and a YAML configuration file.

### Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--with_pepper` | bool | `False` | Enables compatibility with the Pepper robot |
| `--start_cameras` | bool | `False` | Initializes the robot's cameras during module startup |
| `--ia` | bool | `False` | Enables AI-dependent services (face, pose, hand detection, chess) |
| `--llm_mode` | string | `"ollama"` | VLM backend: `"ollama"` or `"openai"` |
| `--vlm_model` | string | `"gemma3_12b"` | Model name for VLM service |
| `--vlm_max_tokens` | int | `500` | Maximum tokens for VLM response |

### YAML Configuration

VLM settings can also be configured via `src/config.yaml`:

```yaml
vlm:
  llm_mode: "ollama"  # Options: "ollama", "openai"
  model: "gemma3_12b" # Model name
  max_tokens: 500
```

Command-line arguments override YAML configuration.

### How to Configure

Configuration is passed through command-line arguments in the format `--<name>[=value]`:

```bash
rosrun vision_utilities vision_utilities.py --with_pepper --start_cameras --ia
```

If a value is not provided, boolean flags are assumed to be `True`.

### Examples

**Basic local setup:**
```bash
rosrun vision_utilities vision_utilities.py
```

**With Pepper robot and AI services:**
```bash
rosrun vision_utilities vision_utilities.py --with_pepper --start_cameras --ia
```

**With OpenAI VLM backend:**
```bash
rosrun vision_utilities vision_utilities.py --llm_mode=openai --vlm_model=GPT-4o
```

## Additional Nodes

### Camera Publisher

Publishes webcam frames for local testing:

```bash
rosrun vision_utilities camera_publisher.py
```

### Polygon Renderer

Visualizes polygon topics on camera feed:

```bash
# With Pepper camera
rosrun vision_utilities visualizer.py

# With local camera
rosrun vision_utilities visualizer.py --local
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GPT_API` | Azure OpenAI API key (required when `llm_mode=openai`) |

## Error Handling

Errors during argument parsing will result in descriptive messages, ensuring users are aware of issues like:
- Incorrect argument format
- Unrecognized or unsupported configuration flags
- Validation errors on provided configuration values

## Dependencies

Core dependencies are listed in `requirements_basic.txt`. AI-related dependencies (MediaPipe, Ultralytics, DEAP) are in `requirements_ai.txt`. Full dependencies are in `all_requirements.txt`.

## License

This repository is proprietary and is intended solely for use by SInfonIA. Unauthorized use, distribution, or modification is prohibited.