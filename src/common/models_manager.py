import os
import pathlib
from ultralytics import YOLO

import constants


def get_yolo_model(name: str) -> YOLO:
    model = YOLO(f"{constants.MODELS_FOLDER}/yolo/{name}.pt")

    return model


def get_mediapipe_path(name: str) -> pathlib.Path:
    return constants.MODELS_FOLDER / f"mediapipe/{name}.task"