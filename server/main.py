from flask import Flask, request
from flask_cors import CORS
from flask import render_template
from fastai.vision.all import *
import torch
from torchvision import transforms
from PIL import Image

#Labeling function required for load_learner to work
def GetLabel(fileName):
  return fileName.split('-')[0]

# learn = load_learner(Path('server/export.pkl')) #Import Model
app = Flask(__name__)
cors = CORS(app) #Request will get blocked otherwise on Localhost

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Load the pre-trained model for CIFAR-100
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    # Define a transformation for your input image (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # VGG models typically expect 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = PILImage.create(request.files['file'])
    img = transform(img)
    img = img.unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    model.eval()
    with torch.no_grad():
        output = model(img)

    # Get the predicted class
    probabilities = torch.nn.functional.softmax(output, dim=1)
    prob, predicted_class = probabilities.max(1)

    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    predicted_label = cifar10_classes[predicted_class.item()]
    print(predicted_label, prob)

    return f'{predicted_label} ({prob[0]*100:.0f}%)'

if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)



