import asyncio
import json

import cv2
import numpy as np
import rospy
import websockets
from cv2.typing import MatLike
from cv_bridge import CvBridge
from perception_msgs.srv import (
    ToggleDetectionTopic,
    ToggleDetectionTopicRequest,
    ToggleDetectionTopicResponse,
)
from sensor_msgs.msg import Image

import constants
from utils import models_manager
from utils.camera_topic import CameraTopic


class COCOObjectDetectionService:
    bridge = CvBridge()
    active = False

    def __init__(self, camera: str):
        self.model_name = "yolo11n"
        self._detect_device()

        print(f"Iniciando servicio de detección en modo: {self.device}")

        if self.device == "npu":
            self.websocket_url = "ws://localhost:5230/ws/detect"
            print(
                f"Configurado para inferencia remota vía WebSocket: {self.websocket_url}"
            )
            self.model = None
        else:
            print("Cargando modelo localmente...")
            self.model = models_manager.get_yolo_model(self.model_name)
            if self.device != "cpu":
                self.model.to(self.device)

        self.image_pub = rospy.Publisher(
            constants.TOPIC_COCO_DETECTIONS, Image, queue_size=10
        )
        self.service = rospy.Service(
            constants.SERVICE_DETECT_COCO_OBJECTS,
            ToggleDetectionTopic,
            self.handle_coco_object_detection,
        )
        self.camera = CameraTopic(camera)
        self.sid = None

    def _detect_device(self):
        try:
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

    def camera_subscriber(self, image: MatLike):
        try:
            if self.device == "npu":
                annotated_image = asyncio.run(self.process_via_websocket(image))
            else:
                results = self.model(image)
                annotated_image = results[0].plot()

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
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
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

    def handle_coco_object_detection(self, req: ToggleDetectionTopicRequest):
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
