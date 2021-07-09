import cv2
import mediapipe
import numpy as np


class Draw:
    def __init__(self, frame_size: tuple[int, int]) -> None:
        self.canvas = np.zeros(frame_size, dtype=np.uint16)
        self.frameWidth, self.frameHeight = frame_size

    def draw_message(self, frame: np.ndarray, message: str) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize, = cv2.getTextSize(
            text=message,
            fontFace=font, 
            fontScale=1, 
            thickness=2
        )

        textX = (self.frameWidth - textsize[0]) // 2
        textY = (self.frameHeight + textsize[1]) // 2

        frame = cv2.putText(
            img=frame,
            text=message,
            org=(textX, textY),
            fontFace=font,
            fontScale=1,
            color=(251, 251, 251),
            thickness=2,
        )

    def draw_on_frame(self, frame: np.ndarray):
        pass

    def __merge_canvas(self, frame: np.ndarray) -> np.ndarray:
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv_canvas = cv2.threshold(
            gray_canvas, 
            thresh=50, 
            maxval=255, 
            type=cv2.THRESH_BINARY_INV
        )
        
        inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv_canvas)
        frame = cv2.bitwise_or(frame, self.canvas)
        
        return frame
