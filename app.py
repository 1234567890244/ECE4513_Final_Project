from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import cv2
import os
import numpy as np
import random
from io import BytesIO
from werkzeug.utils import secure_filename
from datetime import datetime
from core.face_utils import FaceUtils
from core.emotion_detector import EmotionDetector
from core.meme_generator import MemeGenerator
from core.text_generator import EmotionFusionGenerator
from core.image_database import create_database
from core.DatabaseManager import DatabaseManager
from core.main import generate_meme

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

fonts = ["SourceHanSansSC-Heavy.otf", "Bian.otf", "Jianjian.otf"]
font_path = "assets/fonts/" + fonts[random.randint(0, 2)]
predictor_path = "core/assets/models/shape_predictor_68_face_landmarks.dat"
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    print("DEEPSEEK_API_KEY Environment Variable not found.")
else:
    efg = EmotionFusionGenerator(api_key)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename, ext = os.path.splitext(secure_filename(filename))
    return f"{basename}_{timestamp}{ext}"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file chosen'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File format not supported'}), 400

    try:
        unique_filename = generate_unique_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(upload_path)

        abs_upload_path = os.path.abspath(upload_path)

        print(abs_upload_path)

        if abs_upload_path is None:
            return jsonify({'error': 'Failed to read image'}), 500
        print("image read successfully")

        meme, output_path = generate_meme(
            abs_upload_path,
            font_path=font_path,
            predictor_path=predictor_path,
        )

        meme_img = np.array(meme)
        meme_img = cv2.cvtColor(meme_img, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".jpg", meme_img)
        if not is_success:
            return jsonify({'error': 'Failed to generate image'}), 500

        io_buf = BytesIO(buffer)
        io_buf.seek(0)

        processed_filename = f"meme_{unique_filename}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, meme_img)

        return jsonify({
            'original': f"/static/uploads/{unique_filename}",
            'processed': f"/static/processed/{processed_filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)