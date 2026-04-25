import pathlib
import urllib.request

from ultralytics import YOLO

import constants


_MEDIAPIPE_MODEL_URLS = {
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def get_yolo_model(name: str) -> YOLO:
    local_model = constants.MODELS_FOLDER / "yolo" / f"{name}.pt"
    model = YOLO(str(local_model)) if local_model.exists() else YOLO(name)

    return model


def get_mediapipe_path(name: str) -> pathlib.Path:
    model_path = constants.MODELS_FOLDER / f"mediapipe/{name}.task"
    if model_path.exists():
        return model_path

    url = _MEDIAPIPE_MODEL_URLS.get(name)
    if url is None:
        raise FileNotFoundError(
            f"Mediapipe model '{name}' not found at {model_path} and no download URL is configured."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Failed to download mediapipe model '{name}' to {model_path}."
        )

    return model_path
