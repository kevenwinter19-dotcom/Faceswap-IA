import torch
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class SimSwapFaceSwapper:
    def __init__(self, device="cuda"):
        self.device = device

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)

    def get_embedding(self, img):
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return faces[0].normed_embedding

    def swap_faces(self, source_frame, target_frame):
        source_embedding = self.get_embedding(source_frame)
        if source_embedding is None:
            return target_frame

        faces = self.app.get(target_frame)
        result = target_frame.copy()

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            face_crop = target_frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Simulação avançada (placeholder melhorado)
            face_crop = cv2.GaussianBlur(face_crop, (15, 15), 0)

            result[y1:y2, x1:x2] = face_crop

        return result
