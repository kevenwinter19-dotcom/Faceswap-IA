import cv2
import numpy as np

class TemporalSmoother:
    def __init__(self):
        self.prev_frame = None

    def smooth_sequence(self, frames):
        smoothed = []

        for frame in frames:
            if self.prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                h, w = flow.shape[:2]
                map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

                map_x = (map_x + flow[:, :, 0]).astype(np.float32)
                map_y = (map_y + flow[:, :, 1]).astype(np.float32)

                frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

            smoothed.append(frame)
            self.prev_frame = frame.copy()

        return smoothed
