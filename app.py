from flask import Flask, request, render_template, jsonify
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
from io import BytesIO
from ultralytics import YOLO  # Ensure this is correctly installed

app = Flask(__name__)

# Define transformations for the classifier
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("Home.html")

@app.route('/about')
def about():
    return render_template("About.html")

@app.route('/contact')
def contact():
    return render_template("Contact.html")

@app.route('/classify')
def classifier():
    return render_template("Classifier.html")

@app.route('/detect')
def detector():
    return render_template("Detector.html")

# Load the classifier model
def load_classifier_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 100)  # Assuming 100 classes
    )
    model.load_state_dict(torch.load('classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

classifier_model = load_classifier_model()

# Load the detector model
def load_detector_model():
    model = YOLO('detector.pt', device='cpu')  # Adjust based on your actual saved model
    model.eval()
    return model

detector_model = load_detector_model()

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        files = request.files.getlist('file')
        results = []
        for file in files:
            if file:
                image = Image.open(file.stream)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_tensor = classifier_transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = classifier_model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    results.append(predicted.item())
        return render_template('Classifier.html', classification_results=results)
    return render_template('Classifier.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        files = request.files.getlist('file')
        results = []
        for file in files:
            if file:
                image = Image.open(file.stream)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                results.append(detector_model(image).pandas().xyxy[0].to_dict(orient='records'))  # Using pandas for easier manipulation
        return render_template('Detector.html', detection_results=results)
    return render_template('Detector.html')




if __name__ == '__main__':
    import os
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)