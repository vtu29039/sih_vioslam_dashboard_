import threading
import time

import av
import cv2

from src.dynafusion_engine import DynaFusionEngine
from src.telemetry_controller import TelemetryController


class VIOProcessor:
    """Runs DynaFusion on every frame streamed from the user's browser camera."""

    def __init__(self):
        self.engine = DynaFusionEngine()
        self.controller = TelemetryController()
        self._lock = threading.Lock()
        self._mask = None
        self._telemetry = None
        self._latency_ms = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        t0 = time.time()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        annotated, mask, _static, dyn_pixels, cx = self.engine.process_frame(img, gray)
        telemetry = self.controller.compute_telemetry(dyn_pixels, cx, frame_width=img.shape[1])
        latency = round((time.time() - t0) * 1000, 1)

        with self._lock:
            self._mask = mask
            self._telemetry = telemetry
            self._latency_ms = latency

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def snapshot(self):
        with self._lock:
            return self._mask, self._telemetry, self._latency_ms
