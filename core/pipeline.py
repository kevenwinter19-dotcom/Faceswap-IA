import cv2
from pathlib import Path
import shutil

from .extract import FrameExtractor
from .swap import SimSwapFaceSwapper
from .smooth import TemporalSmoother
from .render import VideoRenderer

class FaceSwapPipeline:
    def __init__(self, device="cuda"):
        self.extractor = FrameExtractor()
        self.swapper = SimSwapFaceSwapper(device)
        self.smoother = TemporalSmoother()
        self.renderer = VideoRenderer()

    def process_video(self, video_path: Path, face_image_path: Path, output_path: Path):
        temp_dir = Path("data/temp_frames")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        print("Extraindo frames...")
        self.extractor.extract_frames(video_path, temp_dir)

        frame_paths = sorted(temp_dir.glob("*.png"))
        source_img = cv2.imread(str(face_image_path))

        processed_frames = []

        print("Aplicando face swap...")
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            result = self.swapper.swap_faces(source_img, frame)
            processed_frames.append(result)

        print("Suavizando...")
        smoothed = self.smoother.smooth_sequence(processed_frames)

        print("Renderizando...")
        self.renderer.frames_to_video(smoothed, output_path, video_path)

        print("Finalizado:", output_path)
