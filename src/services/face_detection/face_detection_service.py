import os
import sys
import sqlite3
import json
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from deepface import DeepFace
from collections import Counter
from threading import Event
from sensor_msgs.msg import Image
from cv2.typing import MatLike

import constants
from config import VisionModuleConfiguration
from utils.camera_topic import CameraTopic
from perception_msgs.srv import (
    ToggleDetectionTopic,
    ToggleDetectionTopicRequest,
    ToggleDetectionTopicResponse,
    save_face_srv,
    save_face_srvResponse,
    remove_faces_data_srv,
    remove_faces_data_srvResponse,
    get_labels_srv,
    get_labels_srvResponse,
    recognize_face_srv,
    recognize_face_srvResponse
)


class FaceDetectionService:
    bridge = CvBridge()
    active = False



    def __init__(self, camera: str, config: VisionModuleConfiguration):
        self.config = config
        current_dir = os.path.dirname(os.path.realpath(__file__))
        src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        self.db_path = os.path.join(current_dir, "data/facial_recognition.db")
        self._init_db()
        self.model_name = "Facenet512"
        self.distance_threshold = 15
        self.save_count = 0
        self.save_name = ""
        self.save_interval = 30
        self.frame_counter = 0
        self.detection_interval = 1

        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.service = rospy.Service(
            constants.SERVICE_FACE_RECOGNITION,
            ToggleDetectionTopic,
            self.handle_face_recognition_toggle,
        )
        self.save_service = rospy.Service(
            constants.TOPIC_FACE_RECOGNITION+"/save_face",
            save_face_srv,
            self.handle_save_face,
        )

        self.get_service = rospy.Service(
            constants.TOPIC_FACE_RECOGNITION+"/get_face",
            recognize_face_srv,
            self.handle_get_face,
        )

        self.clear_service = rospy.Service(
            constants.TOPIC_FACE_RECOGNITION+"/clear_face_data",
            remove_faces_data_srv,
            self.handle_clear_embeddings,
        )
        self.count_service = rospy.Service(
            constants.TOPIC_FACE_RECOGNITION+"/get_faces_saved",
            get_labels_srv,
            self.handle_get_names_count,
        )

        self.image_pub = None
        if "face_recognition" in config.publish_visualizations:
            self.image_pub = rospy.Publisher(
                constants.TOPIC_FACE_RECOGNITION+"/face_recognition_image", Image, queue_size=10
            )

        self.camera = CameraTopic(camera)
        self.sid = None

    def _init_db(self):
        _, conn = self.connect_db()
        conn.close()

    def connect_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                embedding TEXT
            )
        """)
        conn.commit()
        return cursor, conn

    def get_face_closest_to_center(self, faces_data, frame):
        if not faces_data:
            return None
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2
        closest_face = None
        min_distance = float('inf')
        for face_data in faces_data:
            facial_area = face_data['facial_area']
            face_center_x = facial_area['x'] + facial_area['w'] / 2
            face_center_y = facial_area['y'] + facial_area['h'] / 2
            distance = ((face_center_x - center_x) ** 2 + (face_center_y - center_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_face = face_data
        return closest_face

    def identify_face(self, face_roi):
        try:
            if face_roi.max() <= 1.0:
                face_roi = (face_roi * 255).astype('uint8')

            target_embedding = DeepFace.represent(
                img_path=face_roi, model_name=self.model_name, detector_backend="skip"
            )[0]["embedding"]

            cursor, conn = self.connect_db()
            cursor.execute("SELECT name, embedding FROM embeddings")
            rows = cursor.fetchall()
            
            matches = []
            target_vec = np.array(target_embedding)
            for db_name, embedding_json in rows:
                db_vec = np.array(json.loads(embedding_json))
                distance = np.linalg.norm(db_vec - target_vec)
                if distance < self.distance_threshold:
                    matches.append(db_name)
            conn.close()

            if not matches:
                return "Unknown"

            base_names = [name.rsplit('_', 1)[0] if '_' in name and name.rsplit('_', 1)[1].isdigit() else name for name in matches]
            return Counter(base_names).most_common(1)[0][0]
        except Exception as e:
            rospy.logerr(f"Identification error: {e}")
            return "Error"

    def save_face_from_roi(self, face_roi, name):
        try:
            if face_roi.max() <= 1.0:
                face_roi = (face_roi * 255).astype('uint8')

            embedding = DeepFace.represent(
                img_path=face_roi, model_name=self.model_name, detector_backend="skip"
            )[0]["embedding"]

            cursor, conn = self.connect_db()
            cursor.execute("INSERT INTO embeddings (name, embedding) VALUES (?, ?)", (name, json.dumps(embedding)))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            rospy.logerr(f"Save error: {e}")
            return False

    def camera_subscriber(self, image: MatLike):
        try:
            faces_data = DeepFace.extract_faces(image, enforce_detection=False)
        except Exception as e:
            rospy.logerr(f"DeepFace extraction failed: {e}")
            return

        center_face = self.get_face_closest_to_center(faces_data, image)

        if self.save_count > 0:
            self.frame_counter += self.detection_interval
            if self.frame_counter >= self.save_interval:
                if center_face and center_face.get('confidence', 0) > 0.8:
                    if self.save_face_from_roi(center_face['face'], self.save_name):
                        self.save_count -= 1
                        self.frame_counter = 0
                        rospy.loginfo(f"Capturing face for {self.save_name}. Remaining: {self.save_count}")

        if self.image_pub is not None:
            annotated_frame = image.copy()
            for face_data in faces_data:
                if face_data.get('confidence', 0) < 0.8:
                    continue
                
                area = face_data['facial_area']
                
                if self.save_count > 0 and face_data is center_face:
                    label = f"Saving {self.save_name}... ({self.save_count} left)"
                    color = (0, 255, 255) 
                else:
                    name = self.identify_face(face_data['face'])
                    label = name
                    color = (0, 255, 0) 
                    if name == "Unknown":
                        color = (0, 0, 255) 
                
                cv2.rectangle(annotated_frame, (area['x'], area['y']), 
                              (area['x'] + area['w'], area['y'] + area['h']), color, 2)

                cv2.putText(annotated_frame, label, (area['x'], area['y'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.image_pub.publish(msg)

    def handle_face_recognition_toggle(self, req: ToggleDetectionTopicRequest):
        response = ToggleDetectionTopicResponse()
        if req.state:
            if self.active and self.sid is not None:
                self.camera.unsubscribe(self.sid)
            frames_interval = max(1, req.frames_interval)
            self.active = True
            self.detection_interval = frames_interval
            self.sid = self.camera.subscribe(
                self.camera_subscriber, wait_turns=self.detection_interval
            )
            response.state = "Activated"
        else:
            self.active = False
            if self.sid is not None:
                self.camera.unsubscribe(self.sid)
                self.sid = None
            response.state = "Deactivated"
        return response

    def handle_save_face(self, req):
        if not req.name:
            rospy.logwarn("Face Detection: Save face requested with empty name.")
            return save_face_srvResponse(False)

        if self.save_count > 0:
            rospy.logwarn(f"Face Detection: Save face already in progress for {self.save_name}.")
            return save_face_srvResponse(False)

        self.save_name = req.name
        self.save_count = req.num_pics if req.num_pics > 0 else 1
        self.save_interval = req.interval if req.interval > 0 else 30
        self.frame_counter = self.save_interval 
        
        rospy.loginfo(f"Triggered saving for: {self.save_name}, num_pics={self.save_count}, interval={self.save_interval}")
        return save_face_srvResponse(True)

    def handle_get_face(self, req):
        response = recognize_face_srvResponse()

        if self.save_count > 0:
            rospy.logwarn(
                f"Face Detection: Get face requested while saving {self.save_name}."
            )
            response.approved = False
            response.person = ""
            return response

        num_pics = req.num_pics if req.num_pics > 0 else 1
        recognized_names = []
        frames_collected = 0
        finished = Event()

        def collect_face_name(image: MatLike):
            nonlocal frames_collected

            if finished.is_set():
                return

            name = "Unknown"
            try:
                faces_data = DeepFace.extract_faces(image, enforce_detection=False)
                center_face = self.get_face_closest_to_center(faces_data, image)
                if center_face and center_face.get("confidence", 0) > 0.8:
                    name = self.identify_face(center_face["face"])
                    if not name or name == "Error":
                        name = "Unknown"
            except Exception as e:
                rospy.logerr(f"Face Detection: get_face frame processing failed: {e}")

            recognized_names.append(name)
            frames_collected += 1
            if frames_collected >= num_pics:
                finished.set()

        temp_sid = self.camera.subscribe(collect_face_name, wait_turns=1)
        timeout = max(5.0, float(num_pics) * 2.0)
        finished.wait(timeout=timeout)
        self.camera.unsubscribe(temp_sid)

        if not recognized_names:
            response.approved = False
            response.person = "Unknown"
            return response

        counts = Counter(recognized_names)
        best_name = recognized_names[0]
        best_count = counts[best_name]
        for name in recognized_names[1:]:
            if counts[name] > best_count:
                best_name = name
                best_count = counts[name]

        response.approved = True
        response.person = best_name
        return response

    def handle_clear_embeddings(self, req):
        cursor, conn = self.connect_db()
        cursor.execute("DELETE FROM embeddings")
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return remove_faces_data_srvResponse(True)

    def handle_get_names_count(self, req):
        cursor, conn = self.connect_db()
        cursor.execute("SELECT DISTINCT name FROM embeddings")
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        res = get_labels_srvResponse()
        res.labels = ", ".join(names) if names else "None"
        res.approved = True
        return res

if __name__ == "__main__":
    rospy.init_node("face_detection_service_node")

    try:
        camera_topic = rospy.get_param("~camera_topic", "usb_cam/image_raw")
        
        from config import VisionModuleConfiguration
        config = VisionModuleConfiguration() 

        server = FaceDetectionService(camera_topic, config)
        rospy.loginfo("Face Detection Service is initialized and waiting for requests...")

        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Face Detection Service shutting down.")
    except Exception as e:
        rospy.logerr(f"Failed to start Face Detection Service: {e}")