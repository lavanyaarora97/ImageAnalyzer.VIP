from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
from torch import nn

app = Flask(__name__)

# Load the classification model
classifier_model = models.resnet50(pretrained=False)
num_ftrs = classifier_model.fc.in_features
classifier_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1024),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.Dropout(0.2),
    nn.Linear(128, 100)
)
classifier_model.load_state_dict(torch.load('models/classifier.pth'))
classifier_model.eval()

# Load the object detection model
detector_model = torch.load('models/detector.pt')
detector_model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.229, 0.224])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@app.route('/classify', methods=['POST'])
def classify():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'})
    files = request.files.getlist('file')
    predictions = []
    for file in files:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        outputs = classifier_model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append({'class_id': predicted.item()})
    return jsonify(predictions)

@app.route('/detect', methods=['POST'])
def detect():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'})
    files = request.files.getlist('file')
    results = []
    for file in files:
        img_bytes = file.read()
        image_tensor = transform_image(img_bytes)
        with torch.no_grad():
            outputs = detector_model(image_tensor)
        # Assuming outputs are bounding boxes [x1, y1, x2, y2, score, class]
        detections = []
        for detection in outputs[0]:
            x1, y1, x2, y2, score, class_id = detection
            detections.append({
                'x1': x1.item(),
                'y1': y1.item(),
                'x2': x2.item(),
                'y2': y2.item(),
                'score': score.item(),
                'class_id': class_id.item()
            })
        results.append(detections)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)