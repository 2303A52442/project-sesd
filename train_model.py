import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

DATASET = "dataset"
BATCH = 16
EPOCHS = 10
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("models", exist_ok=True)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
train_data = datasets.ImageFolder(DATASET, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)

# Load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

print("Training started...")

for epoch in range(EPOCHS):
    total, correct, loss_sum = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", colour="green")

    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100*correct/total)

    print(f"Epoch {epoch+1}: Loss={loss_sum/len(train_loader):.4f}, Accuracy={100*correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), "models/kidney_stone_model.pth")
print("Model saved in models/kidney_stone_model.pth")
