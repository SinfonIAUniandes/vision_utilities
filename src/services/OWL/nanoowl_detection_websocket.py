import asyncio
import json

import cv2
import rospy
import websockets
from cv2.typing import MatLike
from cv_bridge import CvBridge
from perception_msgs.srv import (
    PromptObjectDetection,
    PromptObjectDetectionRequest,
    PromptObjectDetectionResponse,
    ToggleDetectionTopic,
    ToggleDetectionTopicRequest,
    ToggleDetectionTopicResponse,
)
from perception_msgs.msg import get_labels_msg
from sensor_msgs.msg import Image

import constants
from config import VisionModuleConfiguration
from utils.camera_topic import CameraTopic


class NanoOWLObjectDetectionService:
    bridge = CvBridge()
    active = False

    def __init__(self, camera: str, config: VisionModuleConfiguration):
        self.prompt = "[a person][a mug][a bottle]"
        self.websocket_url = "ws://localhost:5231/ws/detect"
        self.config = config
        print(f"Configurado para inferencia remota vía WebSocket: {self.websocket_url}")

        self.image_pub = None
        if "owl_detections" in config.publish_visualizations:
            self.image_pub = rospy.Publisher(
                constants.TOPIC_OWL_DETECTIONS, Image, queue_size=10
            )
        self.bboxes_pub = None
        if "owl_bboxes" in config.publish_data:
            self.bboxes_pub = rospy.Publisher(
                constants.TOPIC_OWL_BBOXES, get_labels_msg, queue_size=10
            )
        self.start_topic_service = rospy.Service(
            constants.SERVICE_DETECT_OWL_OBJECTS,
            ToggleDetectionTopic,
            self.handle_owl_object_detection,
        )
        self.prompt_service = rospy.Service(
            constants.SERVICE_DETECT_OWL_OBJECTS + "_prompt",
            PromptObjectDetection,
            self.handle_prompt_object_detection,
        )
        self.camera = CameraTopic(camera)
        self.sid = None

    def handle_owl_object_detection(self, req: ToggleDetectionTopicRequest):
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

    def handle_prompt_object_detection(self, req: PromptObjectDetectionRequest):
        response = PromptObjectDetectionResponse()
        self.prompt = req.prompt
        response.success = True
        return response

    def camera_subscriber(self, image: MatLike):
        try:
            annotated_image, bboxes_data = asyncio.run(self.process_via_websocket(image))
            if annotated_image is not None and self.image_pub is not None:
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated_image, encoding="bgr8"
                )
                self.image_pub.publish(annotated_msg)
            
            if bboxes_data is not None and self.bboxes_pub is not None:
                self.bboxes_pub.publish(bboxes_data)
        except Exception as e:
            rospy.logerr(f"Error en el procesamiento de imagen: {e}")

    async def process_via_websocket(self, frame):
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                await websocket.send(self.prompt)
                await websocket.recv()
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                _, buffer = cv2.imencode(".jpg", frame, encode_param)
                await websocket.send(buffer.tobytes())
                response = await websocket.recv()
                data = json.loads(response)
                annotated_frame = self.draw_detections(frame, data)
                bboxes_data = self._extract_bboxes_from_websocket(data)
                return annotated_frame, bboxes_data
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            rospy.logwarn_throttle(5, f"No se pudo conectar al WebSocket: {e}")
            return frame, None
        except Exception as e:
            rospy.logerr(f"Error en websocket logic: {e}")
            return frame, None

    def draw_detections(self, frame, data):
        frame = frame.copy()  # Make a copy to avoid modifying the original shared frame
        for det in data.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            confidence = det["confidence"]
            class_id = det["class"]
            label = f"{class_id} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(
                frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return frame

    def _extract_bboxes_from_websocket(self, data):
        """Extract bounding boxes from websocket response and create message."""
        msg = get_labels_msg()
        detections = data.get("detections", [])
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            msg.labels.append(str(det.get("class", "unknown")))
            msg.x_coordinates.append(float(x1))
            msg.y_coordinates.append(float(y1))
            msg.widths.append(float(width))
            msg.heights.append(float(height))
            msg.ids.append(float(det.get("id", -1)))
        
        return msg if detections else None
