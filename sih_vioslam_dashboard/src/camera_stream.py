import cv2
import yaml

from src.paths import CONFIG_PATH


class CameraStream:
    """Server-side webcam capture (only for running on the vehicle / your own machine)."""

    def __init__(self, config_path=CONFIG_PATH, index=0):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["camera"]

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])
        self.cap.set(cv2.CAP_PROP_FPS, self.config["fps"])

    def is_open(self):
        return self.cap.isOpened()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, gray

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
