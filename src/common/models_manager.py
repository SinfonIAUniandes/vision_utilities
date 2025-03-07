import os
import pathlib
from ultralytics import YOLO


def get_yolo_model(name: str) -> YOLO:
    models_folder = pathlib.Path(__file__).resolve().parent / "../../models"
    model = YOLO(f"{models_folder}/yolo/{name}.pt")

    return model
