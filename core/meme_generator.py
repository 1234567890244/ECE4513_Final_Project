from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import math


class MemeGenerator:
    def __init__(self, font_path):
        self.font_path = font_path
        self.base_font_size = None
        self.font = None
        self.small_font = None
        self._load_fonts(placeholder_size=30)

    def _load_fonts(self, placeholder_size=30):
        print("loading fonts...")
        try:
            if self.font_path and os.path.exists(self.font_path):
                self.font = ImageFont.truetype(self.font_path, placeholder_size)
                self.small_font = ImageFont.truetype(self.font_path, int(placeholder_size * 0.7))
            else:
                raise FileNotFoundError("error in meme_generator-load_fonts")
        except Exception as e:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    # def _get_dominant_color(self, image_array, k=1):
    #     print("getting dominant color...")
    #     h, w, _ = image_array.shape
    #     scale = min(1.0, 200.0 / max(h, w))
    #     small_img = cv2.resize(image_array, (0, 0), fx=scale, fy=scale)
    #
    #     pixels = small_img.reshape((-1, 3))
    #
    #     kmeans = KMeans(n_clusters=k, n_init=10)
    #     kmeans.fit(pixels)
    #
    #     counts = Counter(kmeans.labels_)
    #     dominant_color = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
    #
    #     return tuple(dominant_color.astype(int))

    def _get_dominant_color(self, image_array, k=1):
        print("getting dominant color from corners...")
        h, w, _ = image_array.shape

        corner_size = int(min(h, w) * 0.3)

        corners = [
            (0, 0, corner_size, corner_size),
            (w - corner_size, 0, w, corner_size),
            (0, h - corner_size, corner_size, h),
            (w - corner_size, h - corner_size, w, h)
        ]

        all_pixels = []

        for corner in corners:
            x1, y1, x2, y2 = corner
            corner_img = image_array[y1:y2, x1:x2]

            scale = min(1.0, 200.0 / max(corner_img.shape[0], corner_img.shape[1]))
            small_img = cv2.resize(corner_img, (0, 0), fx=scale, fy=scale)
            pixels = small_img.reshape((-1, 3))
            all_pixels.append(pixels)

        all_pixels = np.vstack(all_pixels)

        if len(all_pixels) > 0:
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(all_pixels)

            counts = Counter(kmeans.labels_)
            dominant_color = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
            return tuple(dominant_color.astype(int))
        else:
            return (128, 128, 128)

    def _get_contrast_color(self, bg_color):
        print("getting contrast color...")
        r, g, b = [c for c in bg_color]
        luminance = 0.2126 * r / 255.0 + 0.7152 * g / 255.0 + 0.0722 * b / 255.0
        color = (255 - r, 255 - g, 255 - b)

        return (0, 0, 0) if luminance > 0.7 else color

    def _get_complexity_score(self, image_array):
        print("getting complexity score...")
        std_dev = np.std(image_array, axis=(0, 1))
        complexity = np.mean(std_dev)
        return complexity

    def _find_text_region(self, image_array, face, face_rect, landmarks, corner=False):
        print("finding text region...")
        print("=" * 50)
        print("face region: ", face.left(), face.top(), face.right(), face.bottom())
        print("image region: ", face_rect["left"], face_rect["top"], face_rect["right"], face_rect["bottom"])
        print("=" * 50)

        pil_img = Image.fromarray(image_array)

        img_width, img_height = pil_img.size

        safe_left = face_rect["left"]
        safe_top = face_rect["top"]
        safe_right = face_rect["right"]
        safe_bottom = face_rect["bottom"]

        regions = [
            {"name": "top", "rect": (safe_left, safe_top, safe_right, face.top())},
            {"name": "bottom", "rect": (safe_left, face.bottom(), safe_right, safe_bottom)},
            {"name": "left", "rect": (safe_left, safe_top, face.left(), safe_bottom)},
            {"name": "right", "rect": (face.right(), safe_top, safe_right, safe_bottom)}
        ]

        if corner:
            regions = [
                {"name": "top", "rect": (safe_left, safe_top, safe_right, face.top())},
                {"name": "bottom", "rect": (safe_left, safe_top, safe_right, face.bottom())},
                {"name": "left_bottom", "rect": (safe_left, landmarks[48][1], landmarks[48][0], safe_bottom)},
                {"name": "left_top", "rect": (safe_left, safe_top, landmarks[45][0], landmarks[45][1])},
                {"name": "right_bottom", "rect": (landmarks[54][0], landmarks[54][1], safe_right, safe_bottom)},
                {"name": "right_top", "rect": (landmarks[36][0], safe_top, safe_right, landmarks[36][1])},
            ]

        face_x = (face.left() + face.right()) // 2
        face_y = (face.top() + face.bottom()) // 2

        best_region = None
        best_score = -1

        print("=" * 50)
        for region in regions:
            x1, y1, x2, y2 = region["rect"]
            x3, y3, x4, y4 = safe_left, safe_top, safe_right, safe_bottom

            if region["name"] == "top":
                y1 = y3 = max(0, y1-(safe_bottom-face.bottom())//2)
                y4 = y3 + (safe_bottom-safe_top)
                region_img = image_array[y1:y2, x1:x2]
            elif region["name"] == "bottom":
                y2 = y4 = min(img_height, y2+(face.top()-safe_top)//2)
                y3 = y4 - (safe_bottom-safe_top)
                region_img = image_array[y1:y2, x1:x2]
            elif region["name"] == "left":
                x1 = x3 = max(0, x1-(safe_right-face.right())//2)
                x4 = x3 + (safe_right-safe_left)
                region_img = image_array[y1:y2, x1:x2]
            else:
                x2 = x4 = min(img_width, x2+(face.left()-safe_left)//2)
                x3 = x4 - (safe_right-safe_left)
                region_img = image_array[y1:y2, x1:x2]

            complexity = self._get_complexity_score(region_img)
            print(region["name"], complexity)

            region_center_x = (x1 + x2) // 2
            region_center_y = (y1 + y2) // 2
            distance = math.sqrt((face_x - region_center_x) ** 2 + (face_y - region_center_y) ** 2)

            score = 1.0 / (complexity + 1)

            if score > best_score:
                best_score = score
                best_region = {
                    "name": region["name"],
                    "rect": (x1, y1, x2, y2),
                    "center": (region_center_x, region_center_y),
                    "complexity": complexity,
                    "distance": distance,
                    "image": region_img,
                    "safe_region": (x3, y3, x4, y4)
                }

        if best_region is None:
            print("using default region...")
            bottom_region = {
                "name": "bottom",
                "rect": (safe_left, int((safe_top + safe_bottom) * 0.7), safe_right, safe_bottom),
                "center": ((safe_left + safe_right) // 2, int((safe_top + safe_bottom) * 0.85)),
                "complexity": self._get_complexity_score(
                    image_array[safe_top:int((safe_top + safe_bottom) * 0.7), safe_left:safe_right]
                ),
                "image": image_array[safe_top:int((safe_top + safe_bottom) * 0.7), safe_left:safe_right],
                "safe_region": (safe_left, safe_top, safe_right, safe_bottom)
            }
            print("=" * 50)
            return bottom_region

        print("best region: ", best_region["name"])
        print("=" * 50)

        return best_region

    def _draw_vertical_text(self, draw, text, position, font, fill, outline_color, spacing=10):
        x, y = position
        for char in text:
            if outline_color:
                offsets = [(dx, dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2) if dx != 0 or dy != 0]
                for dx, dy in offsets:
                    draw.text((x + dx, y + dy), char, font=font, fill=outline_color, anchor='lt')

            draw.text((x, y), char, font=font, fill=fill, anchor='lt')
            y += font.size

    def _draw_horizontal_text(self, draw, text, position, font, fill, outline_color):
        x, y = position
        if outline_color:
            offsets = [(dx, dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2) if dx != 0 or dy != 0]
            for dx, dy in offsets:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color, anchor='lt')

        draw.text((x, y), text, font=font, fill=fill, anchor='lt')

    def _draw_text(self, draw, text, region, position, font, text_color, outline_color):
        x1, y1, x2, y2 = region["rect"]

        if x2-x1 < y2-y1:
            self._draw_vertical_text(
                draw, text,
                position=position,
                font=font,
                fill=text_color,
                outline_color=outline_color
            )
        else:
            self._draw_horizontal_text(
                draw, text,
                position=position,
                font=font,
                fill=text_color,
                outline_color=outline_color
            )

    def _draw_text_with_outline(self, draw, region, position, text, font, text_color, outline_color):
        print("drawing text with outline...")
        self._draw_text(draw, text, region, position, font=font, text_color=None, outline_color=outline_color)
        self._draw_text(draw, text, region, position, font=font, text_color=text_color, outline_color=None)

    def _draw_text_with_shadow(self, draw, region, position, text, font, text_color, shadow_color):
        print("drawing text with shadow...")
        x, y = position
        self._draw_text(draw, text, region, (x + 2, y + 2), font=font, text_color=shadow_color, outline_color=None)
        self._draw_text(draw, text, region, position, font=font, text_color=text_color, outline_color=None)

    def create_error_meme(self, image, text="no human face detected..."):
        print("creating error meme...")
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        width, height = pil_img.size

        square_size = min(width, height)
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        cropped_img = pil_img.crop((left, top, left + square_size, top + square_size))

        enhancer = ImageEnhance.Contrast(cropped_img)
        cropped_img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(cropped_img)
        cropped_img = enhancer.enhance(1.1)

        img_array = np.array(cropped_img)
        dominant_color = self._get_dominant_color(img_array)
        text_color = self._get_contrast_color(dominant_color)

        draw = ImageDraw.Draw(cropped_img)
        font_size = max(20, square_size // 15)

        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        text_width = draw.textlength(text, font=font)
        text_x = (square_size - text_width) // 2
        text_y = square_size * 0.85

        outline_color = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
        offsets = [(dx, dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2) if dx != 0 or dy != 0]
        for dx, dy in offsets:
            draw.text((text_x + dx, text_y + dy), text, font=font, fill=outline_color)

        draw.text((text_x, text_y), text, font=font, fill=text_color)

        return cropped_img

    def create_meme(self, image, face, landmarks, text):
        print("creating meme...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_img)

        img_width, img_height = pil_img.size

        max_size = min(img_width, img_height)

        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()

        space = max(face_width, face_height)

        square_size = min(max_size, space * 2)
        center_x = face.left() + face_width // 2
        center_y = face.top() + face_height // 2

        start_x = max(0, center_x - square_size // 2)
        start_y = max(0, center_y - square_size // 2)
        end_x = min(img_width, start_x + square_size)
        end_y = min(img_height, start_y + square_size)

        face_rect = {
            'left': start_x,
            'right': end_x,
            'top': start_y,
            'bottom': end_y
        }

        if len(text) <= 2:
            text_region = self._find_text_region(image_rgb, face, face_rect, landmarks, True)
        else:
            text_region = self._find_text_region(image_rgb, face, face_rect, landmarks, False)

        if text_region:
            region_x1, region_y1, region_x2, region_y2 = text_region["rect"]
            region_center_x, region_center_y = text_region["center"]
            print("text_region: ", region_x1, region_y1, region_x2, region_y2)
        else:
            print("no text region")
            safe_left = max(0, face.left() - space // 2)
            safe_top = max(0, face.top() - space // 2)
            safe_right = min(img_width, face.right() + space // 2)
            safe_bottom = min(img_height, face.bottom() + space // 2)

            text_region = {
                "name": "default_bottom",
                "rect": (safe_left, int((safe_top + safe_bottom) * 0.7), safe_right, safe_bottom),
                "center": ((safe_left + safe_right) // 2, int((safe_top + safe_bottom) * 0.85)),
                "complexity": 0,
                "image": image_rgb[safe_top:safe_bottom, safe_left:safe_right]
            }
            region_x1, region_y1, region_x2, region_y2 = text_region["rect"]
            region_center_x, region_center_y = text_region["center"]

        region_img = image_rgb[region_y1:region_y2, region_x1:region_x2]
        region_dominant_color = self._get_dominant_color(region_img)
        text_color = self._get_contrast_color(region_dominant_color)

        print("=" * 50)
        print("region_dominant_color: ", region_dominant_color)
        print("text_color: ", text_color)
        print("=" * 50)

        font_size = 10
        font = ImageFont.truetype(self.font_path, font_size)

        region_width = region_x2 - region_x1
        region_height = region_y2 - region_y1

        while True:
            if text_region["name"] in ['left', 'right']:
                max_char_width = max([draw.textlength(char, font=font) for char in text])
                total_height = len(text) * font_size + (len(text) - 1) * 10

                if max_char_width > region_width * 0.9 or total_height > region_height * 0.9:
                    break
            else:
                text_width = draw.textlength(text, font=font)
                if text_width >= region_width * 0.9 or font_size >= region_height * 0.9:
                    break

            font_size += 1
            font = ImageFont.truetype(self.font_path, font_size)

        if text_region["name"] in ['left', 'right']:
            max_char_width = max([draw.textlength(char, font=font) for char in text])
            total_height = len(text) * font_size

            x = region_x1 + (region_width - max_char_width) // 2
            y = region_y1 + (region_height - total_height) // 2

        else:
            text_width = draw.textlength(text, font=font)
            text_height = font_size

            x = region_center_x - text_width // 2
            y = region_y1 + (region_height - text_height) // 2

        outline_color = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
        shadow_color = (0, 0, 0, 128) if text_color == (255, 255, 255) else (255, 255, 255, 128)

        complexity = text_region["complexity"]
        if complexity > 80:
            self._draw_text_with_outline(draw, text_region, (x, y), text, font, text_color, outline_color)
        elif complexity > 60:
            self._draw_text_with_shadow(draw, text_region, (x, y), text, font, text_color, shadow_color)
        else:
            self._draw_text(draw, text, text_region, (x, y), font=font, text_color=text_color, outline_color=None)

        start_x, start_y, end_x, end_y = text_region["safe_region"]

        # '''====================== [red] third face regctangle [red] ======================'''
        # draw.rectangle(
        #     (face.left(), face.top(), face.right(), face.bottom()),
        #     outline="red",
        #     width=3
        # )
        # '''====================== [red] third face regctangle [red] ======================'''

        cropped_img = pil_img.crop((start_x, start_y, end_x, end_y))
        return cropped_img
