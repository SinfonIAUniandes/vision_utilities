import time
from typing import Dict

import cv2
import numpy as np
import rospy
from perception_msgs.srv import (
    detect_chess_srv,
    detect_chess_srvRequest,
    detect_chess_srvResponse,
)

import constants
from services.chess_detection import board_detection, pieces_detection
from utils.camera_topic import CameraTopic


class ChessDetection:
    def __init__(self, camera: str):
        rospy.Service(constants.SERVICE_DETECT_CHESS, detect_chess_srv, self.callback)
        self.camera = CameraTopic(camera)

    def callback(self, request: detect_chess_srvRequest):
        response = detect_chess_srvResponse()
        response.board = []
        response.pieces = []
        response.fen = ""

        start = time.time()
        to_process = None

        while time.time() - start < 2:
            image = self.camera.get_image()
            if image is not None:
                to_process = image
                break
        if to_process is None:
            return response

        to_process = cv2.resize(to_process, (640, 640))

        board_corners = board_detection.get_corners(to_process)
        if board_corners is None:
            return response

        response.board = board_corners.flatten().tolist()
        board_corners = np.array(board_corners)

        def reorder_corners(pts):
            centroid = np.mean(pts, axis=0)
            ordered_pts = sorted(
                pts, key=lambda p: (np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
            )
            return np.array(ordered_pts, dtype=np.int32)

        board_corners = reorder_corners(board_corners)
        target_perspective = np.array([[40, 40], [600, 40], [600, 600], [40, 600]])
        transformer = cv2.getPerspectiveTransform(
            target_perspective.astype(np.float32), board_corners.astype(np.float32)
        )

        cells = {}
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
        margin = 40
        width = 640 - margin * 2
        gap = width / 8

        for i in range(8):
            for j in range(8):
                id = f"{letters[i]}{j + 1}"
                x = margin + gap * i
                y = margin + gap * j
                cells[id] = [
                    [x, y],
                    [x, y + gap],
                    [x + gap, y + gap],
                    [x + gap, y],
                ]

        for cell, poly in cells.items():
            poly = np.array([poly], dtype=np.float32)
            poly = cv2.perspectiveTransform(poly, transformer).tolist()
            poly = [list(map(round, p)) for p in poly[0]]
            cells[cell] = poly

        pieces_result = pieces_detection.get_predictions(to_process)
        boxes = pieces_result.boxes.xywh.cpu().numpy()
        boxes = [list(map(round, box)) for box in boxes]
        confidence = pieces_result.boxes.conf.cpu().numpy().tolist()
        classes = pieces_result.boxes.cls.cpu().numpy().tolist()

        pieces_by_name: Dict[int, str] = {
            3: "white_pawn",
            9: "black_pawn",
            5: "white_rook",
            11: "black_rook",
            2: "white_knight",
            8: "black_knight",
            0: "white_bishop",
            6: "black_bishop",
            1: "white_king",
            7: "black_king",
            4: "white_queen",
            10: "black_queen",
        }

        pieces_by_cell = {}

        for i in range(len(boxes)):
            box = boxes[i]
            x, y, w, h = box
            w = w // 2
            h = h // 2
            base_at = y + h - w

            if cv2.pointPolygonTest(np.array([board_corners]), (x, base_at), False) < 0:
                continue
            elif confidence[i] < 0.4:
                continue

            class_id = int(classes[i])
            if class_id not in pieces_by_name:
                continue
            piece_name = pieces_by_name[class_id]

            for cell, polygon in cells.items():
                if cv2.pointPolygonTest(np.array([polygon]), (x, base_at), False) > 0:
                    if pieces_by_cell.get(cell) is None:
                        pieces_by_cell[cell] = []
                    pieces_by_cell[cell].append((piece_name, confidence[i]))

        for cell, pieces in pieces_by_cell.items():
            pieces_by_cell[cell] = sorted(pieces, key=lambda x: x[1])

        def dictionary_to_fen(position_dict):
            piece_to_fen = {
                "white_king": "K",
                "white_queen": "Q",
                "white_rook": "R",
                "white_bishop": "B",
                "white_knight": "N",
                "white_pawn": "P",
                "black_king": "k",
                "black_queen": "q",
                "black_rook": "r",
                "black_bishop": "b",
                "black_knight": "n",
                "black_pawn": "p",
            }
            board = [["" for _ in range(8)] for _ in range(8)]

            for cell, piece_name in position_dict.items():
                if piece_name in piece_to_fen:
                    file = cell[0]
                    rank = cell[1]
                    row = 8 - int(rank)
                    col = ord(file) - ord("a")
                    board[row][col] = piece_to_fen[piece_name]

            fen_rows = []
            for row in board:
                empty_count = 0
                fen_row = ""
                for square in row:
                    if square == "":
                        empty_count += 1
                    else:
                        if empty_count > 0:
                            fen_row += str(empty_count)
                            empty_count = 0
                        fen_row += square
                if empty_count > 0:
                    fen_row += str(empty_count)
                fen_rows.append(fen_row)

            return "/".join(fen_rows)

        for cell, piece in pieces_by_cell.items():
            pieces_by_cell[cell] = max(piece, key=lambda x: x[1])[0]

        response.fen = dictionary_to_fen(pieces_by_cell)
        return response
