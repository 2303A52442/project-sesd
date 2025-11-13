import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "models/kidney_stone_model.pth"
CLASS_NAMES = ["Normal", "Stone"]

# Load model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        score, pred = torch.max(probs, 1)

    return CLASS_NAMES[pred.item()], float(score.item())



