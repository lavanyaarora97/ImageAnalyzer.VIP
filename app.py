from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

model_yolo = DetectMultiBackend('yolov5scoco8', device=DEVICE, img_size=IMG_SIZE)
model_yolo.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))
model_cifar = torch.load('cifar100_model.pkl')  # Adjust path as needed
model_cifar.eval()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process the image for classification
            return redirect(url_for('result', filename=filename, type='classify'))
    return render_template('classify.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process the image for detection
            return redirect(url_for('result', filename=filename, type='detect'))
    return render_template('detect.html')

@app.route('/result/<filename>/<type>')
def result(filename, type):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Placeholder: Actual image processing and model inference logic here
    return send_from_directory(directory=app.config['UPLOAD_FOLDER'], filename=filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
