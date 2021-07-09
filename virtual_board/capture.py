import cv2
import numpy as np


class Capture:
    def __init__(self) -> None:
        self.capture = cv2.VideoCapture(0)
        self.frameWidth = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def get_frame_size(self):
        return self.frameWidth, self.frameHeight

    def get_frame(self) -> tuple[np.ndarray, int, int]:
        success, frame = self.capture.read()
        if not success:
            raise "Camera not Found"

        frame = cv2.flip(frame, 1)
        return frame

    def show_frame(self, frame: np.ndarray) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Result", frame)

    def release(self) -> None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = Capture()
    while True:
        frame = cap.video_start()
        cap.show_frame(frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
