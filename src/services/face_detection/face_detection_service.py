import sqlite3
import json
import numpy as np
import cv2
from deepface import DeepFace
from collections import Counter

# Available models:
# "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
# "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet", "Buffalo_L"
MODEL_NAME = "Facenet512" 
DISTANCE_THRESHOLD = 15  #This is based on the model 
DB_PATH = "data/facial_recognition.db"



def connect_db():
    conn = sqlite3.connect(DB_PATH)
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

#Finds nearest face to center 
def get_face_closest_to_center(faces_data, frame):
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


#Receives a frame and identifies the face in the center
def identify_face_in_frame(frame):

    try:
        faces_data = DeepFace.extract_faces(frame, enforce_detection=False)
        center_face = get_face_closest_to_center(faces_data, frame)
        
        if not center_face or center_face.get('confidence', 0) < 0.8:
            return None

        face_roi = center_face['face']
        if face_roi.max() <= 1.0:
            face_roi = (face_roi * 255).astype('uint8')

        target_embedding = DeepFace.represent(img_path=face_roi, model_name=MODEL_NAME, detector_backend="skip")[0]["embedding"]

        cursor, conn = connect_db()
        cursor.execute("SELECT name, embedding FROM embeddings")
        rows = cursor.fetchall()
        
        matches = []
        for db_name, embedding_json in rows:
            db_vec = np.array(json.loads(embedding_json))
            distance = np.linalg.norm(db_vec - np.array(target_embedding))
            if distance < DISTANCE_THRESHOLD:
                matches.append(db_name)
        
        conn.close()
        if not matches: return "Unknown"

        base_names = [name.rsplit('_', 1)[0] if '_' in name and name.rsplit('_', 1)[1].isdigit() else name for name in matches]
        return Counter(base_names).most_common(1)[0][0]
    except Exception as e:
        print(f"Identification error: {e}")
        return None

#Receives a frame and savves de face closest to center 
def save_face_from_frame(frame, name):
    try:
        faces_data = DeepFace.extract_faces(frame, enforce_detection=False)
        center_face = get_face_closest_to_center(faces_data, frame)
        if not center_face: return False

        face_roi = center_face['face']
        if face_roi.max() <= 1.0:
            face_roi = (face_roi * 255).astype('uint8')

        embedding = DeepFace.represent(img_path=face_roi, model_name=MODEL_NAME, detector_backend="skip")[0]["embedding"]

        cursor, conn = connect_db()
        cursor.execute("INSERT INTO embeddings (name, embedding) VALUES (?, ?)", (name, json.dumps(embedding)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Save error: {e}"); return False

#Clears all embbedings (this is specially usefull for changing models since embeddings change)
def clear_all_embeddings():
    cursor, conn = connect_db(); cursor.execute("DELETE FROM embeddings")
    count = cursor.rowcount; conn.commit(); conn.close(); return count

#Gets embedding count from db
def get_embedding_count():
    cursor, conn = connect_db(); cursor.execute("SELECT COUNT(DISTINCT name) FROM embeddings")
    count = cursor.fetchone()[0]; conn.close(); return count