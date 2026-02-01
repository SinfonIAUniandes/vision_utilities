from typing import List

from cv2.typing import MatLike
from ultralytics.engine.results import Results

from utils import models_manager


def get_predictions(image: MatLike):
    model = models_manager.get_yolo_model("chess_pieces")

    results: List[Results] = model.predict(source=image, save=False, device="cpu")

    return results[0]
