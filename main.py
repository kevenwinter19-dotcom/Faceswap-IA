from pathlib import Path
from core.pipeline import FaceSwapPipeline

if __name__ == "__main__":
    pipeline = FaceSwapPipeline()

    video = Path("data/input/video.mp4")
    face = Path("data/input/face.png")
    output = Path("data/output/result.mp4")

    pipeline.process_video(video, face, output)
