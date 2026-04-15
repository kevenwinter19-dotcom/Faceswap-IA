import cv2
from pathlib import Path

class VideoRenderer:
    def frames_to_video(self, frames, output_path: Path, ref_video: Path):
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(str(output_path),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              30, (w, h))

        for f in frames:
            out.write(f)

        out.release()
