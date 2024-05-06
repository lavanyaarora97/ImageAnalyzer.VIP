from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import models, transforms
from PIL import Image
import io


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

# Load models (ensure correct paths and configurations)
classifier_model = torch.load('models/classifier.pth', map_location=torch.device('cpu'))
detector_model = torch.load('models/detector.pt', map_location=torch.device('cpu'))

# Define transforms
classifier_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Adjust according to your specific model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Handle multiple files
        results = []
        for file in files:
            if file:
                image = Image.open(file.stream)
                image_tensor = classifier_transform(image).unsqueeze(0)
                outputs = classifier_model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                results.append(predicted.item())
        return render_template('Classifier.html', classification_results=results)

    return render_template('Classifier.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Handle multiple files
        results = []
        for file in files:
            if file:
                image = Image.open(file.stream)
                image_tensor = transforms.functional.to_tensor(image).unsqueeze(0)  # Update your transform if needed
                outputs = detector_model(image_tensor)  # Adjust based on actual model usage
                results.append(outputs)
        return render_template('Detector.html', detection_results=results)

    return render_template('Detector.html')

if __name__ == '__main__':
    import os
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)