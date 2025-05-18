from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io

app = Flask(__name__)

# Load model
class EfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetV2Classifier, self).__init__()
        self.efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.efficientnet(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetV2Classifier(num_classes=4).to(device)
model.load_state_dict(torch.load("lesion_classifier.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels
class_names = ["NV", "BCC", "SCC", "Melanoma"]

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400

    img = Image.open(request.files['image']).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        label = class_names[predicted.item()]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
