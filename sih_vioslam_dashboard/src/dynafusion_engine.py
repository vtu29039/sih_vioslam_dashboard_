import cv2
import numpy as np
import yaml

from src.paths import CONFIG_PATH


class DynaFusionEngine:
    """Dynamic feature segmentation and feature filtering (DynaFusion-VIO demo)."""

    def __init__(self, config_path=CONFIG_PATH):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)["dynafusion"]
        self.prev_gray = None
        self._kernel = np.ones((3, 3), np.uint8)

    def process_frame(self, frame, gray):
        h, w = gray.shape[:2]
        mask_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        annotated_frame = frame.copy()

        static_features = []
        dynamic_pixels = 0
        motion_centroid_x = w // 2

        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)  # suppress sensor noise

        if self.prev_gray is not None and self.prev_gray.shape == gray_blur.shape:
            diff = cv2.absdiff(gray_blur, self.prev_gray)
            _, motion_mask = cv2.threshold(
                diff, self.cfg["optical_flow_diff_thresh"], 255, cv2.THRESH_BINARY
            )
            # remove speckle, then close small holes in moving regions
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self._kernel)
            motion_mask = cv2.dilate(motion_mask, self._kernel, iterations=1)

            mask_display[motion_mask == 255] = [0, 0, 255]
            dynamic_pixels = int(np.count_nonzero(motion_mask))

            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.cfg["max_corners"],
                qualityLevel=self.cfg["quality_level"],
                minDistance=self.cfg["min_distance"],
            )

            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    ix = min(max(int(x), 0), w - 1)
                    iy = min(max(int(y), 0), h - 1)
                    if motion_mask[iy, ix] == 0:
                        static_features.append((ix, iy))
                        cv2.circle(annotated_frame, (ix, iy), 3, (0, 255, 0), -1)
                    else:
                        cv2.circle(annotated_frame, (ix, iy), 4, (0, 0, 255), 1)

            M = cv2.moments(motion_mask, binaryImage=True)
            if M["m00"] > 0:
                motion_centroid_x = int(M["m10"] / M["m00"])

        self.prev_gray = gray_blur
        return annotated_frame, mask_display, len(static_features), dynamic_pixels, motion_centroid_x
