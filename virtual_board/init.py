import cv2
import numpy as np
from draw import Draw
from capture import Capture
from detect_hands import DetectHands


COMMAND_LIST = {0: 'Nothing', 1: 'Writing', 2: 'Erasing'}


class WhiteBoard:
    def __init__(self, frame_size: tuple[int, int] = None, debug: bool = False) -> None:
        self.draw = None
        self.detect = None
        self.debug = debug
        
        if frame_size is not None:
            self.frameWidth, self.frameHeight = frame_size
            self.draw = Draw(frame_size)
            self.detect = DetectHands(frame_size, debug)
            
    def set_frame_size(self, frame_size: tuple[int, int]):
        self.__init__(frame_size, self.debug)
        
    def __check_frame(self, frame) -> bool:
        return True

    def __get_command(self, fingers: np.ndarray) -> int:
        if "Index" in fingers:
            if "Middle" in fingers:
                if "Ring" in fingers:
                    return 0
                return 2
            return 1
        return 0

    def __execute_error(self, frame: np.ndarray, error: str) -> np.ndarray:
        return self.draw.draw_message(frame, error)

    def __execute_command(self, frame:np.ndarray,  command: int, postions: dict) -> np.ndarray:
        if command == 0:
            return self.draw.draw_on_frame(frame)
        elif command == 1:
            return frame
        elif command == 2:
            return frame

    def add_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.draw is None or self.detect is None:
            raise 'Set Frame Size'
            
        if not self.__check_frame(frame):
            raise 'Invalid Frame'
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, positions = self.detect.detect_hands(frame)

        if not success:
            return self.__execute_error(frame, 'Ensure the whole hand is in the frame')

        if positions:
            fingers = self.detect.get_fingers(positions)
            command = self.__get_command(fingers)
            print(command)
            # return self.__execute_command(frame, command, positions)
            
        return frame

    def start_capture(self) -> None:
        self.capture = Capture()
        self.set_frame_size(self.capture.get_frame_size())

        while True:
            frame = self.capture.get_frame()
            frame = self.add_frame(frame)
            self.capture.show_frame(frame)

            if cv2.waitKey(1) == 27:
                break

        self.capture.release()


if __name__ == "__main__":
    white_board = WhiteBoard(debug=True)
    white_board.start_capture()
