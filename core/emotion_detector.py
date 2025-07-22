import time
import cv2
import numpy as np
from fer import FER
from deepface import DeepFace
from collections import defaultdict


class EmotionDetector:
    def __init__(self, primary_threshold=0.3, secondary_threshold=0.25,
                 max_retries=3, top_n=3, primary_weight=0.5):

        self.primary_detector = FER(mtcnn=False)
        self.secondary_models = ["VGG-Face", "Facenet", "OpenFace"]

        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.max_retries = max_retries
        self.top_n = top_n
        self.primary_weight = primary_weight
        self.secondary_weight = 1 - primary_weight
        self.last_valid_emotions = ["neutral"]
        self.last_valid_percentages = [1.0]

        self.emotion_groups = {
            "positive": ["happy", "surprise"],
            "negative": ["angry", "disgust", "fear", "sad"],
            "neutral": ["neutral"]
        }

        self.emotion_mapping = {
            "angry": "angry", "disgust": "disgust", "fear": "fear",
            "happy": "happy", "sad": "sad", "surprise": "surprise",
            "neutral": "neutral", "angriness": "angry", "sadness": "sad",
            "happiness": "happy", "fearfulness": "fear", "disgusted": "disgust"
        }

        self.group_mapping = {}
        for group, emotions in self.emotion_groups.items():
            for emotion in emotions:
                self.group_mapping[emotion] = group

    def get_emotion_group(self, emotion):
        return self.group_mapping.get(emotion, "neutral")

    def is_group_conflict(self, emotions):
        if len(emotions) <= 1:
            return False

        groups = [self.get_emotion_group(e) for e in emotions]

        non_neutral_groups = set(g for g in groups if g != "neutral")
        return len(non_neutral_groups) > 1

    def resolve_group_conflict(self, emotions, percentages):
        print("resolving emotion conflict...")
        if not self.is_group_conflict(emotions):
            print("no emotion conflict")
            return emotions, percentages

        group_scores = defaultdict(float)
        for e, p in zip(emotions, percentages):
            group = self.get_emotion_group(e)
            if group != "neutral":
                group_scores[group] += p

        best_group = max(group_scores.items(), key=lambda x: x[1])[0]
        print("emotions: ", emotions)
        print("best group: ", best_group)

        filtered = []
        filtered_p = []
        for e, p in zip(emotions, percentages):
            if self.get_emotion_group(e) == best_group or self.get_emotion_group(e) == "neutral":
                filtered.append(e)
                filtered_p.append(p)

        total = sum(filtered_p)
        if total > 0:
            filtered_p = [p / total for p in filtered_p]
        print("filtered emotion conflict: ", filtered)

        return filtered[:self.top_n], filtered_p[:self.top_n]

    def fuse_results(self, primary_emotions, primary_percentages,
                     secondary_emotions, secondary_percentages):
        if not primary_emotions and not secondary_emotions:
            return ["neutral"], [1.0]
        elif not primary_emotions:
            return self.resolve_group_conflict(
                secondary_emotions[:self.top_n],
                secondary_percentages[:self.top_n]
            )
        elif not secondary_emotions:
            return self.resolve_group_conflict(
                primary_emotions[:self.top_n],
                primary_percentages[:self.top_n]
            )

        fused = defaultdict(float)
        group_totals = defaultdict(float)

        for e, p in zip(primary_emotions, primary_percentages):
            std_e = self.emotion_mapping.get(e.lower(), e.lower())
            fused[std_e] += p * self.primary_weight
            group_totals[self.get_emotion_group(std_e)] += p * self.primary_weight

        for e, p in zip(secondary_emotions, secondary_percentages):
            std_e = self.emotion_mapping.get(e.lower(), e.lower())
            fused[std_e] += p * self.secondary_weight
            group_totals[self.get_emotion_group(std_e)] += p * self.secondary_weight

        sorted_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        fused_emotions = []
        fused_percentages = []
        for e, p in sorted_items:
            if p >= 0.2:
                fused_emotions.append(e)
                fused_percentages.append(p)

        print("fused emotions: {}".format(fused_emotions))

        final_emotions, final_percentages = self.resolve_group_conflict(
            fused_emotions, fused_percentages
        )

        total = sum(final_percentages)
        if total > 0:
            final_percentages = [p / total for p in final_percentages]

        return final_emotions, final_percentages

    def preprocess_face(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def detect_with_primary(self, face_roi):
        try:
            processed_face = self.preprocess_face(face_roi)
            results = self.primary_detector.detect_emotions(processed_face)

            if not results:
                return [], []

            selected_region = max(results, key=lambda x: x['box'][2] * x['box'][3])
            emotions = selected_region['emotions']

            standardized = {}
            for e, c in emotions.items():
                std_e = self.emotion_mapping.get(e.lower(), e.lower())
                standardized[std_e] = standardized.get(std_e, 0) + c

            sorted_items = sorted(standardized.items(), key=lambda x: x[1], reverse=True)
            primary_emotions = [e for e, _ in sorted_items]
            primary_percentages = [p for _, p in sorted_items]

            total = sum(primary_percentages)
            if total > 0:
                primary_percentages = [p / total for p in primary_percentages]

            return primary_emotions, primary_percentages

        except Exception as e:
            print(f"Error in primary model detection: {str(e)}")
            return [], []

    def detect_with_secondary(self, face_roi):
        try:
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            combined = defaultdict(float)

            try:
                analysis = DeepFace.analyze(
                    img_path=rgb_face,
                    actions=['emotion'],
                    enforce_detection=True,
                    detector_backend='opencv',
                    silent=True,
                )

                for result in analysis:
                    for e, c in result['emotion'].items():
                        std_e = self.emotion_mapping.get(e.lower(), e.lower())
                        combined[std_e] += c

                sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
                secondary_emotions = [e for e, _ in sorted_items]
                secondary_percentages = [p for _, p in sorted_items]

                total = sum(secondary_percentages)
                if total > 0:
                    secondary_percentages = [p / total for p in secondary_percentages]

                return secondary_emotions, secondary_percentages

            except Exception as e:
                print(f"Error in secondary model: {str(e)}")
                return [], []

        except Exception as e:
            print(f"Error in secondary model detection: {str(e)}")
            return [], []

    def detect_emotion(self, image, face):
        face_roi = image
        self.retry_count = 0

        while self.retry_count < self.max_retries:
            try:
                primary_emotions, primary_percentages = self.detect_with_primary(face_roi)
                secondary_emotions, secondary_percentages = self.detect_with_secondary(face_roi)

                print(f"primary result: {list(zip(primary_emotions, primary_percentages))}")
                print(f"secondary result: {list(zip(secondary_emotions, secondary_percentages))}")

                final_emotions, final_percentages = self.fuse_results(
                    primary_emotions, primary_percentages,
                    secondary_emotions, secondary_percentages
                )

                self.last_valid_emotions = final_emotions
                self.last_valid_percentages = final_percentages

                return final_emotions, final_percentages

            except Exception as e:
                print(f"Error in emotion detection: {str(e)}")
                self.retry_count += 1
                time.sleep(0.1)

        return self.last_valid_emotions, self.last_valid_percentages
