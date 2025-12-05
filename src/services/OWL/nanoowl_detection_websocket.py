import rospy
import cv2
import numpy as np
import asyncio
import websockets
import json
import constants

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.srv import ToggleDetectionTopicRequest, ToggleDetectionTopicResponse, ToggleDetectionTopic, PromptObjectDetection, PromptObjectDetectionRequest, PromptObjectDetectionResponse

class NanoOWLObjectDetectionService:
    bridge = CvBridge()
    image = None
    active = False

    def __init__(self, camera: str):
        self.active = False
        self.prompt = "[a person][a mug][a bottle]"  # ejemplo de prompt

        # URL del WebSocket para inferencia remota
        self.websocket_url = "ws://localhost:5231/ws/detect"
        print(f"Configurado para inferencia remota vía WebSocket: {self.websocket_url}")

        self.image_pub = rospy.Publisher(constants.TOPIC_OWL_DETECTIONS, Image, queue_size=10)
        self.start_topic_service = rospy.Service(constants.SERVICE_DETECT_OWL_OBJECTS, ToggleDetectionTopic, self.handle_owl_object_detection)
        self.prompt_service = rospy.Service(constants.SERVICE_DETECT_OWL_OBJECTS + "_prompt", PromptObjectDetection, self.handle_prompt_object_detection)
        rospy.Subscriber(camera, Image, self.camera_subscriber)

    def handle_owl_object_detection(self, req: ToggleDetectionTopicRequest):
        response = ToggleDetectionTopicResponse()
        if req.state:
            self.active = True
        else:
            self.active = False
            
        response.state = "Active:" + str(self.active)
        return response

    def handle_prompt_object_detection(self, req: PromptObjectDetectionRequest):
        response = PromptObjectDetectionResponse()
        self.prompt = req.prompt
        response.success = True
        return response

    def camera_subscriber(self, msg: Image):
        if not self.active:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            annotated_image = None

            annotated_image = asyncio.run(self.process_via_websocket(cv_image))

            if annotated_image is not None:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                self.image_pub.publish(annotated_msg)

        except Exception as e:
            rospy.logerr(f"Error en el procesamiento de imagen: {e}")

    async def process_via_websocket(self, frame):
        """
        Envía el frame al servidor WebSocket, recibe las detecciones y dibuja sobre el frame.
        """
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                # 1. Enviar el prompt actual
                await websocket.send(self.prompt)
                await websocket.recv()  # Esperar confirmación
                # 2. Enviar la imagen
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                await websocket.send(buffer.tobytes())

                response = await websocket.recv()
                data = json.loads(response)
                
                return self.draw_detections(frame, data)
                
        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError) as e:
            rospy.logwarn_throttle(5, f"No se pudo conectar al WebSocket: {e}")
            return frame # Retorna la imagen sin anotaciones en caso de error
        except Exception as e:
            rospy.logerr(f"Error en websocket logic: {e}")
            return frame

    def draw_detections(self, frame, data):
        """
        Dibuja las cajas y etiquetas basadas en el JSON recibido del WebSocket.
        """
        for det in data.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            confidence = det["confidence"]
            class_id = det["class"]
            
            # Crear etiqueta
            label = f"{class_id} {confidence:.2f}"
            
            # Dibujar rectángulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar fondo para el texto (para mejor legibilidad)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Dibujar texto
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame