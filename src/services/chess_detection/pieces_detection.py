from typing import List
import numpy as np
from cv2.typing import MatLike
import random
from ultralytics.engine.results import Results
from common import models_manager


def get_predictions(image: MatLike):
    model = models_manager.get_yolo_model("chess_pieces")

    results: List[Results] = model.predict(source=image, save=False, device="cpu")

    results[0].save("resultado.jpg")
