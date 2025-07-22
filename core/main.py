import cv2
from core.face_utils import FaceUtils
from core.emotion_detector import EmotionDetector
from core.meme_generator import MemeGenerator
from core.image_database import create_database
from core.DatabaseManager import DatabaseManager
from core.text_generator import EmotionFusionGenerator
from core.text_library import get_random_text
import os
import random


def generate_meme(image_path, font_path=None, output_path=None, predictor_path=None):
    print(f"start creating meme: {image_path}...")

    image = cv2.imread(image_path)
    print("image read...")

    face_utils = FaceUtils(predictor_path)
    print("face utils ready...")
    emotion_detector = EmotionDetector()
    print("emotion detector ready...")
    meme_generator = MemeGenerator(font_path)
    print("meme generator ready...")
    text_generator = EmotionFusionGenerator(predictor_path)
    print("text generator ready...")

    faces = face_utils.detect_faces(image)
    print("faces detected...")

    if not faces:
        print("no human face detected...")
        return meme_generator.create_error_meme("no human faces"), None

    face = faces[0]
    landmarks = face_utils.get_landmarks(image, face)
    emotion, percentage = emotion_detector.detect_emotion(image, face)
    text = text_generator.generate_caption(emotion, percentage)

    if not text:
        text = get_random_text(emotion[0])

    meme_img = meme_generator.create_meme(
        image, face, landmarks, text
    )

    if output_path is None:
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"meme_{os.path.basename(image_path)}")

    meme_img.save(output_path)
    print(f"meme saved to: {output_path}...")

    return meme_img, output_path


if __name__ == "__main__":

    create_database()
    DB_PATH = "dataset/emotions.sqlite"
    db = DatabaseManager(DB_PATH)
    image_id = random.randint(1, 133)
    input_image = db.get_image_by_id(image_id)
    fonts = ["SourceHanSansSC-Heavy.otf", "Bian.otf", "Jianjian.otf"]
    font_path = "assets/fonts/" + fonts[random.randint(0, len(fonts) - 1)]
    predictor_path = "assets/models/shape_predictor_68_face_landmarks.dat"
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("DEEPSEEK_API_KEY Environment Variable not found.")
    else:
        efg = EmotionFusionGenerator(api_key)

    meme, output_path = generate_meme(
        input_image,
        font_path=font_path,
        predictor_path=predictor_path,
    )

    if meme:
        meme.show()
