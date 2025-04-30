import rospy

from perception_msgs.srv import (
    detect_chess_srv,
    detect_chess_srvRequest,
    detect_chess_srvResponse,
)
from sensor_msgs.msg import Image
from robot_toolkit_msgs.msg import vision_tools_msg
from robot_toolkit_msgs.srv import vision_tools_srv
import constants

import cv2
from cv_bridge import CvBridge
import time
import numpy as np

from services.chess_detection import board_detection, pieces_detection


class ChessDetection:
    bridge = CvBridge()
    image = None

    def __init__(self, camera: str):
        rospy.Service(constants.SERVICE_DETECT_CHESS, detect_chess_srv, self.callback)
        rospy.Subscriber(camera, Image, self.camera_subscriber)

    def camera_subscriber(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def callback(self, request: detect_chess_srvRequest):
        response = detect_chess_srvResponse()
        response.board = []
        response.pieces = []
        response.fen = ""


        start = time.time()
        to_process = self.image

        while time.time() - start < 2:
            if self.image is not None:
                to_process = self.image
                break
        if to_process is None:
            return response
        
        to_process = cv2.resize(to_process, (640, 640))

        start_time = time.time()

        board_corners = board_detection.get_corners(to_process)
        response.board = board_corners.flatten().tolist()

        board_corners = np.array(board_corners)
        print(board_corners)

        def reorder_corners(pts):
            centroid = np.mean(pts, axis=0)

            ordered_pts = sorted(pts, key=lambda p: (np.arctan2(p[1] - centroid[1], p[0] - centroid[0])))
            return np.array(ordered_pts, dtype=np.int32)
        
        board_corners = reorder_corners(board_corners)

        target_perspective = np.array([[40, 40], [600, 40], [600, 600], [40, 600]])

        transformer = cv2.getPerspectiveTransform(target_perspective.astype(np.float32), board_corners.astype(np.float32))

        cells = { }

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
                    [x + gap, y]
                ]


        for cell, poly in cells.items():
            poly = np.array([poly], dtype=np.float32)
            poly = cv2.perspectiveTransform(poly, transformer).tolist()
            poly = [list(map(round, p)) for p in poly[0]]
            #cv2.polylines(frame, [np.array(poly, dtype=np.int32)], True, palette[0], 2)
            cells[cell] = poly

        end_time = time.time()
        print("Took %.2fs" % (end_time - start_time))
        
        pieces_result = pieces_detection.get_predictions(to_process)

        boxes = pieces_result.boxes.xywh.cpu().numpy()
        boxes = [list(map(round, box)) for box in boxes]

        confidence = pieces_result.boxes.conf.cpu().numpy().tolist()
        classes = pieces_result.boxes.cls.cpu().numpy().tolist()

        pieces_by_name = {
            "3": "white_pawn",
            "9": "black_pawn",
            "5": "white_rook",
            "11": "black_rook",
            "2": "white_knight",
            "8": "black_knight",
            "0": "white_bishop",
            "6": "black_bishop",
            "1": "white_king",
            "7": "black_king",
            "4": "white_queen",
            "10": "black_queen",
        }

        pieces_by_cell = {}

        for i in range(len(boxes)):
            box = boxes[i]
            x, y, w, h = box
            w = w // 2
            h = h // 2

            base_at = y + h - w

            if cv2.pointPolygonTest(np.array([board_corners]), (x, base_at), False) < 0:
                color = (255, 0, 0)
            elif confidence[i] < 0.4:
                color = (0, 0, 255)
            else:
                color = (0, int(255 * confidence[i]), 0)
            
            piece_name = pieces_by_name[pieces_result.names[int(classes[i])]]


            for cell, polygon in cells.items():
                #print(cell, polygon)
                if cv2.pointPolygonTest(np.array([polygon]), (x, base_at), False) > 0:
                    if pieces_by_cell.get(cell) is None:
                        pieces_by_cell[cell] = []
                    pieces_by_cell[cell].append((piece_name, confidence[i]))

        for cell, pieces in pieces_by_cell.items():
            pieces_by_cell[cell] = sorted(pieces, key=lambda x: x[1])

        print(pieces_by_cell)
    
        def dictionary_to_fen(position_dict):
            # Mapping of piece names to FEN symbols
            piece_to_fen = {
                "white_king": "K", "white_queen": "Q", "white_rook": "R",
                "white_bishop": "B", "white_knight": "N", "white_pawn": "P",
                "black_king": "k", "black_queen": "q", "black_rook": "r",
                "black_bishop": "b", "black_knight": "n", "black_pawn": "p"
            }
            
            # Initialize an 8x8 board with empty squares
            board = [["" for _ in range(8)] for _ in range(8)]
            
            # Place pieces on the board based on the dictionary
            for cell, piece_name in position_dict.items():
                if piece_name in piece_to_fen:
                    file = cell[0]
                    rank = cell[1]
                    row = 8 - int(rank)  # Convert rank to board row (0-indexed)
                    col = ord(file) - ord('a')  # Convert file to board column (0-indexed)
                    board[row][col] = piece_to_fen[piece_name]
            
            # Generate the FEN string row by row
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
            
            # Combine rows with "/" to create the full FEN
            fen = "/".join(fen_rows)
            return fen
            
        for cell, piece in pieces_by_cell.items():
            pieces_by_cell[cell] = max(piece, key=lambda x: x[1])[0]

        response.fen = dictionary_to_fen(pieces_by_cell)

        return response
