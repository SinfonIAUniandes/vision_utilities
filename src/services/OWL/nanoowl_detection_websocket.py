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
from sensor_msgs.msg import Image

import constants
from utils.camera_topic import CameraTopic


class NanoOWLObjectDetectionService:
    bridge = CvBridge()
    active = False

    def __init__(self, camera: str):
        self.prompt = "[a person][a mug][a bottle]"
        self.websocket_url = "ws://localhost:5231/ws/detect"
        print(f"Configurado para inferencia remota vía WebSocket: {self.websocket_url}")

        self.image_pub = rospy.Publisher(
            constants.TOPIC_OWL_DETECTIONS, Image, queue_size=10
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
            annotated_image = asyncio.run(self.process_via_websocket(image))
            if annotated_image is not None:
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated_image, encoding="bgr8"
                )
                self.image_pub.publish(annotated_msg)
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
                return self.draw_detections(frame, data)
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            rospy.logwarn_throttle(5, f"No se pudo conectar al WebSocket: {e}")
            return frame
        except Exception as e:
            rospy.logerr(f"Error en websocket logic: {e}")
            return frame

    def draw_detections(self, frame, data):
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
