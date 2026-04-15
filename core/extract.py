import ffmpeg
from pathlib import Path

class FrameExtractor:
    def extract_frames(self, video_path: Path, output_dir: Path):
        output_dir.mkdir(exist_ok=True)

        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(
            stream,
            str(output_dir / "%06d.png"),
            vf="fps=30",
            format="image2"
        )
        ffmpeg.run(stream, quiet=True)
