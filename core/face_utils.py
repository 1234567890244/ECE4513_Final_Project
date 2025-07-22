import dlib
import cv2
import os
from pathlib import Path


class FaceUtils:
    def __init__(self, predictor_path=None):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None

        base_dir = Path(__file__).resolve().parent.parent

        if predictor_path is None:
            predictor_path = base_dir / "assets" / "models" / "shape_predictor_68_face_landmarks.dat"
        else:
            predictor_path = Path(base_dir / predictor_path)

        predictor_path = predictor_path.resolve()

        if not predictor_path.exists():
            print("Predictor path not found")
            return

        try:
            self.predictor = dlib.shape_predictor(str(predictor_path))
        except Exception as e:
            print(f"Failed to load predictor: {e}")
            import traceback
            traceback.print_exc()

    def detect_faces(self, image):
        faces = self.detector(image, 0)

        # '''====================== [green] first face regctangle [green] ======================'''
        # for face in faces:
        #     x1, y1 = face.left(), face.top()
        #     x2, y2 = face.right(), face.bottom()
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # '''====================== [green] first face regctangle [green] ======================'''

        return faces

    def get_landmarks(self, image, face):
        if self.predictor is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            landmarks = self.predictor(gray, face)
            return [(p.x, p.y) for p in landmarks.parts()]
        except Exception as e:
            print(f"Failed to get landmarks: {e}")
            return []
