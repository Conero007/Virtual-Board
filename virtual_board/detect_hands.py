import mediapipe
import numpy as np


BASE = 0
CHECK_LIST = (
    ("Index", (6, 8)),
    ("Middle", (10, 12)),
    ("Ring", (14, 16)),
    ("Pinky", (18, 20)),
)


class DetectHands:
    def __init__(self, frame_size: tuple[int, int], debug: bool = False) -> None:
        self.debug = debug
        self.frameWidth, self.frameHeight = frame_size

        self.drawingModule = mediapipe.solutions.drawing_utils
        self.handsModule = mediapipe.solutions.hands
        self.hands = self.handsModule.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1,
        )

    def detect_hands(self, frame: np.ndarray) -> tuple[bool, dict]:
        positions = dict()
        results = self.hands.process(frame)

        if results.multi_hand_landmarks is not None:
            (handLandmarks,) = results.multi_hand_landmarks
            if self.debug:
                self.drawingModule.draw_landmarks(frame, handLandmarks, self.handsModule.HAND_CONNECTIONS)
            for ind, point in enumerate(self.handsModule.HandLandmark):
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoordinatesLandmark = (
                    self.drawingModule._normalized_to_pixel_coordinates(
                        normalizedLandmark.x,
                        normalizedLandmark.y,
                        self.frameWidth,
                        self.frameHeight,
                    )
                )

                if pixelCoordinatesLandmark is None:
                    return False, None

                positions[ind] = pixelCoordinatesLandmark
            return True, positions
        return False, None

    def get_fingers(self, positions: dict) -> list[str]:
        fingers = list()
        for finger, coordinate in CHECK_LIST:
            if self.__is_up(positions, coordinate):
                fingers.append(finger)
        return fingers

    def __get_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        distance = abs(x1 - x2) + abs(y1 - y2)
        return distance

    def __is_up(self, positions: dict, finger: np.ndarray) -> bool:
        base = positions[BASE]
        pip = positions[finger[0]]
        tip = positions[finger[1]]

        dist_base_pip = self.__get_distance(*base, *pip)
        dist_base_tip = self.__get_distance(*base, *tip)

        if dist_base_pip > dist_base_tip:
            return False
        return True
